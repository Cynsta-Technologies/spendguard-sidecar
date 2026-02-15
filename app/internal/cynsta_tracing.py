from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:  # pragma: no cover - optional dependency
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

GENESIS_PREV_HASH = "0" * 96


def canonicalize_json(value: Any) -> str:
    return _serialize(value)


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _serialize_float(value)
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize(item) for item in value) + "]"
    if isinstance(value, dict):
        items: list[str] = []
        for key in sorted(value.keys()):
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            items.append(f"{_escape_string(key)}:{_serialize(value[key])}")
        return "{" + ",".join(items) + "}"
    raise TypeError(f"Unsupported type for canonicalization: {type(value).__name__}")


def _escape_string(value: str) -> str:
    parts: list[str] = ['"']
    for char in value:
        code = ord(char)
        if char == '"':
            parts.append('\\"')
        elif char == "\\":
            parts.append("\\\\")
        elif char == "\b":
            parts.append("\\b")
        elif char == "\f":
            parts.append("\\f")
        elif char == "\n":
            parts.append("\\n")
        elif char == "\r":
            parts.append("\\r")
        elif char == "\t":
            parts.append("\\t")
        elif code < 0x20:
            parts.append(f"\\u{code:04x}")
        else:
            parts.append(char)
    parts.append('"')
    return "".join(parts)


def _serialize_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("NaN and Infinity are not permitted in canonical JSON")
    if value == 0.0:
        return "0"

    raw = repr(value).lower()
    if "e" not in raw:
        return _strip_trailing_zeros(raw)

    mantissa, exp_str = raw.split("e", 1)
    exponent = int(exp_str)
    if -6 <= exponent <= 20:
        return _exp_to_fixed(mantissa, exponent)

    mantissa = _strip_trailing_zeros(mantissa)
    sign = "+" if exponent >= 0 else "-"
    return f"{mantissa}e{sign}{abs(exponent)}"


def _strip_trailing_zeros(value: str) -> str:
    if "." not in value:
        return value
    stripped = value.rstrip("0").rstrip(".")
    return stripped or "0"


def _exp_to_fixed(mantissa: str, exponent: int) -> str:
    sign = ""
    if mantissa.startswith("-"):
        sign = "-"
        mantissa = mantissa[1:]
    if "." in mantissa:
        left, right = mantissa.split(".", 1)
        digits = left + right
        decimal_index = len(left)
    else:
        digits = mantissa
        decimal_index = len(digits)

    new_index = decimal_index + exponent
    if new_index <= 0:
        value = "0." + ("0" * (-new_index)) + digits
    elif new_index >= len(digits):
        value = digits + ("0" * (new_index - len(digits)))
    else:
        value = digits[:new_index] + "." + digits[new_index:]

    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return sign + value


def compute_sha384_hex(value: str) -> str:
    return hashlib.sha384(value.encode("utf-8")).hexdigest()


def compute_data_hash(data: Any) -> str:
    return compute_sha384_hex(canonicalize_json(data))


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _hash_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha384(value.encode("utf-8")).hexdigest()


def _join_text(values: list[str] | tuple[str, ...] | None) -> str | None:
    if not values:
        return None
    return "\n".join([item for item in values if item is not None])


@dataclass
class _RunState:
    span: Any
    start_time: float
    tool_name: str | None = None
    tool_args: Any | None = None
    model: str | None = None
    prompt_hash: str | None = None
    prompt_text: str | None = None


class CynstaTracer(BaseCallbackHandler):
    def __init__(
        self,
        *,
        ingest_endpoint: str | None = None,
        api_key: str | None = None,
        mode: str = "server_trust",
        session_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        service_name: str = "cynsta-langchain",
        resource_attrs: dict[str, str] | None = None,
        batch_delay_ms: int | None = None,
        batch_max_size: int | None = None,
        batch_max_queue: int | None = None,
        batch_timeout_ms: int | None = None,
        capture_prompt_response: bool | None = None,
    ) -> None:
        self.mode = mode
        if mode not in {"server_trust", "client_trust"}:
            raise ValueError("mode must be 'server_trust' or 'client_trust'")
        self.hash_source = "client" if mode == "client_trust" else "server"
        self.session_id = session_id or str(uuid.uuid4())
        self.tenant_id = tenant_id
        self.user_id = user_id or os.getenv("CYNSTA_USER_ID")
        self.capture_prompt_response = _resolve_bool_setting(
            "CYNSTA_PROMPT_RESPONSE_ENABLED",
            capture_prompt_response,
        )

        self._sequence_index = 1
        self._prev_hash = GENESIS_PREV_HASH
        self._lock = threading.Lock()
        self._tool_runs: dict[str, _RunState] = {}
        self._llm_runs: dict[str, _RunState] = {}

        endpoint = ingest_endpoint or os.getenv("CYNSTA_INGEST_URL") or "http://localhost:4318/v1/traces"
        api_key = api_key or os.getenv("CYNSTA_INGEST_API_KEY") or os.getenv("INGEST_API_KEY")
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        batch_delay_ms = _resolve_int_setting("CYNSTA_EXPORT_INTERVAL_MS", batch_delay_ms)
        batch_max_size = _resolve_int_setting("CYNSTA_EXPORT_BATCH_SIZE", batch_max_size)
        batch_max_queue = _resolve_int_setting("CYNSTA_EXPORT_QUEUE_SIZE", batch_max_queue)
        batch_timeout_ms = _resolve_int_setting("CYNSTA_EXPORT_TIMEOUT_MS", batch_timeout_ms)
        _ensure_non_negative("batch_delay_ms", batch_delay_ms)
        _ensure_positive("batch_max_size", batch_max_size)
        _ensure_positive("batch_max_queue", batch_max_queue)
        _ensure_non_negative("batch_timeout_ms", batch_timeout_ms)

        resource = Resource.create({"service.name": service_name, **(resource_attrs or {})})
        self._provider = TracerProvider(resource=resource)
        processor_kwargs: dict[str, int] = {}
        if batch_delay_ms is not None:
            processor_kwargs["schedule_delay_millis"] = batch_delay_ms
        if batch_max_size is not None:
            processor_kwargs["max_export_batch_size"] = batch_max_size
        if batch_max_queue is not None:
            processor_kwargs["max_queue_size"] = batch_max_queue
        if batch_timeout_ms is not None:
            processor_kwargs["export_timeout_millis"] = batch_timeout_ms
        self._provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers), **processor_kwargs)
        )
        self._tracer = self._provider.get_tracer("cynsta.langchain")

    def reset_session(self, session_id: str | None = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        with self._lock:
            self._sequence_index = 1
            self._prev_hash = GENESIS_PREV_HASH

    def flush(self) -> None:
        self._provider.force_flush()

    def shutdown(self) -> None:
        self._provider.shutdown()

    def record_tool_usage(
        self,
        tool_name: str,
        tool_args: Any,
        tool_result: Any | None = None,
        *,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        span = self._tracer.start_span(f"tool:{tool_name}")
        data = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result,
            "duration_ms": duration_ms,
            "error": error,
        }
        attributes = {
            "cynsta.tool.name": tool_name,
            "cynsta.tool.args": _json_dumps(tool_args),
            "cynsta.tool.result": _json_dumps(tool_result) if tool_result is not None else None,
            "cynsta.tool.error": error,
            "cynsta.tool.duration_ms": duration_ms,
        }
        self._finalize_span(span, "tool_usage", data, attributes)
        span.end()

    def record_llm_interaction(
        self,
        *,
        model: str,
        prompt: str | None,
        completion: str | None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> None:
        span = self._tracer.start_span(f"llm:{model}")
        data = {
            "model": model,
            "prompt_hash": _hash_text(prompt),
            "completion_hash": _hash_text(completion),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        if self.capture_prompt_response:
            if prompt is not None:
                data["prompt"] = prompt
            if completion is not None:
                data["completion"] = completion
        attributes = {
            "cynsta.llm.model": model,
            "cynsta.llm.prompt_hash": data["prompt_hash"],
            "cynsta.llm.completion_hash": data["completion_hash"],
            "cynsta.llm.tokens_in": tokens_in,
            "cynsta.llm.tokens_out": tokens_out,
        }
        if self.capture_prompt_response:
            if prompt is not None:
                attributes["cynsta.llm.prompt"] = prompt
            if completion is not None:
                attributes["cynsta.llm.completion"] = completion
        self._finalize_span(span, "llm_interaction", data, attributes)
        span.end()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = _extract_tool_name(serialized, kwargs)
        tool_args = _parse_json(kwargs.get("inputs") or input_str or kwargs.get("input"))
        span = self._tracer.start_span(f"tool:{tool_name or 'tool'}")
        if run_id:
            self._tool_runs[run_id] = _RunState(
                span=span,
                start_time=time.time(),
                tool_name=tool_name,
                tool_args=tool_args,
            )
        else:
            data = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": None,
                "duration_ms": None,
                "error": None,
            }
            self._finalize_span(span, "tool_usage", data, {})
            span.end()

    def on_tool_end(self, output: Any, run_id: str | None = None, **kwargs: Any) -> None:
        self._close_tool_run(run_id, output=output, error=None)

    def on_tool_error(self, error: Exception | str, run_id: str | None = None, **kwargs: Any) -> None:
        self._close_tool_run(run_id, output=None, error=str(error))

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        model = _extract_model(serialized, kwargs)
        prompt_text = _join_text(prompts)
        span = self._tracer.start_span(f"llm:{model or 'llm'}")
        if run_id:
            self._llm_runs[run_id] = _RunState(
                span=span,
                start_time=time.time(),
                model=model,
                prompt_hash=_hash_text(prompt_text),
                prompt_text=prompt_text if self.capture_prompt_response else None,
            )
        else:
            data = {
                "model": model,
                "prompt_hash": _hash_text(prompt_text),
                "completion_hash": None,
                "tokens_in": None,
                "tokens_out": None,
            }
            attributes = {
                "cynsta.llm.model": model,
                "cynsta.llm.prompt_hash": data["prompt_hash"],
                "cynsta.llm.completion_hash": None,
                "cynsta.llm.tokens_in": None,
                "cynsta.llm.tokens_out": None,
            }
            if self.capture_prompt_response and prompt_text is not None:
                data["prompt"] = prompt_text
                attributes["cynsta.llm.prompt"] = prompt_text
            self._finalize_span(span, "llm_interaction", data, attributes)
            span.end()

    def on_llm_end(self, response: Any, run_id: str | None = None, **kwargs: Any) -> None:
        completion_text, tokens_in, tokens_out = _extract_llm_response(response)
        self._close_llm_run(run_id, completion_text, tokens_in, tokens_out)

    def on_llm_error(self, error: Exception | str, run_id: str | None = None, **kwargs: Any) -> None:
        self._close_llm_run(run_id, completion_text=None, tokens_in=None, tokens_out=None, error=str(error))

    def _close_tool_run(self, run_id: str | None, output: Any, error: str | None) -> None:
        if not run_id or run_id not in self._tool_runs:
            return
        run = self._tool_runs.pop(run_id)
        duration_ms = int((time.time() - run.start_time) * 1000)
        data = {
            "tool_name": run.tool_name,
            "tool_args": run.tool_args,
            "tool_result": _parse_json(output),
            "duration_ms": duration_ms,
            "error": error,
        }
        attributes = {
            "cynsta.tool.name": run.tool_name,
            "cynsta.tool.args": _json_dumps(run.tool_args) if run.tool_args is not None else None,
            "cynsta.tool.result": _json_dumps(data["tool_result"]) if data["tool_result"] is not None else None,
            "cynsta.tool.error": error,
            "cynsta.tool.duration_ms": duration_ms,
        }
        self._finalize_span(run.span, "tool_usage", data, attributes)
        run.span.end()

    def _close_llm_run(
        self,
        run_id: str | None,
        completion_text: str | None,
        tokens_in: int | None,
        tokens_out: int | None,
        error: str | None = None,
    ) -> None:
        if not run_id or run_id not in self._llm_runs:
            return
        run = self._llm_runs.pop(run_id)
        data = {
            "model": run.model,
            "prompt_hash": run.prompt_hash,
            "completion_hash": _hash_text(completion_text) if error is None else None,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        attributes = {
            "cynsta.llm.model": run.model,
            "cynsta.llm.prompt_hash": run.prompt_hash,
            "cynsta.llm.completion_hash": data["completion_hash"],
            "cynsta.llm.tokens_in": tokens_in,
            "cynsta.llm.tokens_out": tokens_out,
            "cynsta.llm.error": error,
        }
        if self.capture_prompt_response:
            if run.prompt_text is not None:
                data["prompt"] = run.prompt_text
                attributes["cynsta.llm.prompt"] = run.prompt_text
            if completion_text is not None:
                data["completion"] = completion_text
                attributes["cynsta.llm.completion"] = completion_text
        self._finalize_span(run.span, "llm_interaction", data, attributes)
        run.span.end()

    def _finalize_span(
        self,
        span: Any,
        event_type: str,
        data: dict[str, Any],
        extra_attributes: dict[str, Any],
    ) -> None:
        span.set_attribute("cynsta.event_type", event_type)
        span.set_attribute("cynsta.session_id", self.session_id)
        span.set_attribute("cynsta.hash_source", self.hash_source)
        if self.tenant_id:
            span.set_attribute("cynsta.tenant_id", self.tenant_id)
        if self.user_id:
            span.set_attribute("cynsta.user_id", self.user_id)

        for key, value in extra_attributes.items():
            if value is not None:
                span.set_attribute(key, value)

        if self.user_id and "user_id" not in data:
            data["user_id"] = self.user_id

        span.set_attribute("cynsta.data", canonicalize_json(data))

        if self.hash_source == "client":
            with self._lock:
                sequence_index = self._sequence_index
                prev_hash = self._prev_hash
                data_hash = compute_data_hash(data)
                self._sequence_index += 1
                self._prev_hash = data_hash
            span.set_attribute("cynsta.sequence_index", sequence_index)
            span.set_attribute("cynsta.prev_hash", prev_hash)
            span.set_attribute("cynsta.data_hash", data_hash)


def _extract_tool_name(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
    if isinstance(serialized, dict):
        name = serialized.get("name") or serialized.get("id")
        if isinstance(name, str):
            return name
        if isinstance(name, list) and name:
            return str(name[-1])
    tool_name = kwargs.get("name") or kwargs.get("tool_name")
    if isinstance(tool_name, str):
        return tool_name
    return None


def _extract_model(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
    if isinstance(serialized, dict):
        name = serialized.get("name") or serialized.get("id")
        if isinstance(name, str):
            return name
        if isinstance(name, list) and name:
            return str(name[-1])
    model = kwargs.get("model") or kwargs.get("model_name")
    if isinstance(model, str):
        return model
    return None


def _extract_llm_response(response: Any) -> tuple[str | None, int | None, int | None]:
    completion_text = None
    tokens_in = None
    tokens_out = None

    if response is None:
        return completion_text, tokens_in, tokens_out

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, dict):
            tokens_in = _safe_int(token_usage.get("prompt_tokens"))
            tokens_out = _safe_int(token_usage.get("completion_tokens"))

    generations = getattr(response, "generations", None)
    if generations:
        texts: list[str] = []
        for gen_list in generations:
            for gen in gen_list:
                text = getattr(gen, "text", None)
                if text is not None:
                    texts.append(text)
        completion_text = _join_text(texts)

    return completion_text, tokens_in, tokens_out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_int_setting(env_var: str, value: int | None) -> int | None:
    if value is not None:
        return value
    raw = os.getenv(env_var)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer") from exc


def _resolve_bool_setting(env_var: str, value: bool | None) -> bool:
    if value is not None:
        return value
    raw = os.getenv(env_var)
    if raw is None or raw == "":
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_positive(name: str, value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ensure_non_negative(name: str, value: int | None) -> None:
    if value is None:
        return
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
