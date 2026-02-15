from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from pathlib import Path

from app.core.store import BudgetError, CapStore, store_from_env
from app.billing import apply_context_cliff_to_rates, compute_cost_breakdown
from app.evidence import EvidenceEmitter
from app.pricing import RateTableProvider, cost_cents, estimate_tokens_text
from app.providers.anthropic_provider import (
    call_anthropic_messages,
    extract_anthropic_completion,
    extract_anthropic_usage,
)
from app.providers.gemini_provider import (
    call_gemini_generate_content,
    extract_gemini_completion,
    extract_gemini_usage,
)
from app.providers.openai_provider import (
    call_openai_chat,
    call_openai_responses,
    clamp_openai_max_output_tokens,
    clamp_openai_max_tokens,
    extract_openai_usage,
)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load_dotenv() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dotenv_path = root / ".env"
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception as exc:
        raise RuntimeError("python-dotenv is required to load .env") from exc
    load_dotenv(dotenv_path=dotenv_path)


_load_dotenv()

app = FastAPI(title="Cynsta SpendGuard", version="0.1.0")
store: CapStore = store_from_env()
rate_provider = RateTableProvider.from_env()
rate_provider.force_refresh()
evidence = EvidenceEmitter()


def _allowed_origins() -> list[str]:
    raw = os.getenv("CAP_ALLOWED_ORIGINS", "")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _mode() -> str:
    # Sidecar wrapper is intentionally local/BYOK only.
    return "sidecar"


def require_org(x_api_key: Optional[str] = Header(None)) -> str:
    _ = x_api_key
    return "local"


def _resolve_provider_key(provider: str) -> str:
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required")
        return key
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is required")
        return key
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is required")
        return key
    raise HTTPException(status_code=400, detail="provider must be openai, gemini, or anthropic")


def _model_rates(provider: str, model: str) -> tuple[int, int]:
    card = rate_provider.get_rate_card(provider, model)
    if card is None:
        raise HTTPException(status_code=400, detail=f"Unknown model for pricing: {provider}:{model}")
    return card.input_cents_per_1m, card.output_cents_per_1m


def _rate_card(provider: str, model: str):
    card = rate_provider.get_rate_card(provider, model)
    if card is None:
        raise HTTPException(status_code=400, detail=f"Unknown model for pricing: {provider}:{model}")
    return card


def _estimate_openai_input_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens_text(content)
        elif isinstance(content, list):
            # Multimodal support would need modality-aware pricing and tokenization.
            raise HTTPException(status_code=400, detail="Multimodal content is not supported in MVP")
    return total + 8 * max(1, len(messages))


def _estimate_openai_responses_input_tokens(payload: dict[str, Any]) -> int:
    inp = payload.get("input")
    if isinstance(inp, str):
        return estimate_tokens_text(inp)
    if isinstance(inp, list):
        # Best-effort estimate for chat-style input arrays.
        # We only support plain text content in MVP.
        total = 0
        for item in inp:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                total += estimate_tokens_text(content)
            elif isinstance(content, list):
                raise HTTPException(status_code=400, detail="Multimodal content is not supported in MVP")
        return total + 8 * max(1, len(inp))
    raise HTTPException(status_code=400, detail="input is required (string or list)")


def _estimate_anthropic_input_tokens(system: str | None, messages: list[dict[str, Any]]) -> int:
    total = estimate_tokens_text(system or "")
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens_text(content)
        elif isinstance(content, list):
            # Anthropic supports content blocks; MVP only supports plain text content.
            raise HTTPException(status_code=400, detail="Anthropic content blocks are not supported in MVP")
    return total + 8 * max(1, len(messages))

def _anthropic_rate_card_for_accounting(card):
    # Keep cliff semantics consistent between preflight WCEC and settlement.
    if card.context_cliff_threshold_tokens is not None:
        return card
    return type(card)(
        **{
            **card.__dict__,
            "context_cliff_threshold_tokens": 200_000,
            "context_cliff_input_multiplier": 2.0,
            "context_cliff_output_multiplier": 1.5,
        }
    )


@app.post("/v1/agents")
def create_agent(payload: dict[str, Any], organization_id: str = Depends(require_org)) -> dict[str, str]:
    name = payload.get("name") if isinstance(payload, dict) else None
    if name is not None and not isinstance(name, str):
        raise HTTPException(status_code=400, detail="name must be a string")
    agent_id = store.create_agent(organization_id=organization_id, name=name)
    return {"agent_id": agent_id}


@app.get("/v1/agents")
def list_agents(organization_id: str = Depends(require_org)) -> dict[str, Any]:
    agents = store.list_agents(organization_id=organization_id)
    return {
        "agents": [
            {
                "agent_id": row.agent_id,
                "name": row.name,
                "created_at": row.created_at,
            }
            for row in agents
        ]
    }


@app.post("/v1/agents/{agent_id}/budget")
def set_budget(agent_id: str, payload: dict[str, Any], organization_id: str = Depends(require_org)) -> dict[str, Any]:
    hard_limit_cents = payload.get("hard_limit_cents")
    topup_cents = payload.get("topup_cents", 0)
    if not isinstance(hard_limit_cents, int):
        raise HTTPException(status_code=400, detail="hard_limit_cents is required (int)")
    if not isinstance(topup_cents, int):
        raise HTTPException(status_code=400, detail="topup_cents must be an int")
    try:
        budget = store.upsert_budget(organization_id, agent_id, hard_limit_cents, topup_cents)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return budget.__dict__


@app.get("/v1/agents/{agent_id}/budget")
def get_budget(agent_id: str, organization_id: str = Depends(require_org)) -> dict[str, Any]:
    try:
        budget = store.get_budget(organization_id, agent_id)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return budget.__dict__


@app.post("/v1/agents/{agent_id}/runs")
def create_run(agent_id: str, organization_id: str = Depends(require_org)) -> dict[str, str]:
    # Runs are tracked via budget locks + usage ledger for MVP.
    _ = organization_id
    _ = agent_id
    return {"run_id": str(uuid.uuid4())}


@app.post("/v1/agents/{agent_id}/runs/{run_id}/openai/chat/completions")
def openai_chat_completions(
    agent_id: str,
    run_id: str,
    payload: dict[str, Any],
    organization_id: str = Depends(require_org),
) -> Any:
    return _openai_chat_impl(
        organization_id=organization_id,
        agent_id=agent_id,
        run_id=run_id,
        payload=payload,
    )


@app.post("/v1/chat/completions")
def openai_compat_chat_completions(
    payload: dict[str, Any],
    x_cynsta_agent_id: Optional[str] = Header(None),
    x_cynsta_run_id: Optional[str] = Header(None),
    organization_id: str = Depends(require_org),
) -> Any:
    agent_id = x_cynsta_agent_id
    if not agent_id:
        raise HTTPException(status_code=400, detail="Missing x-cynsta-agent-id header")
    run_id = x_cynsta_run_id or str(uuid.uuid4())
    return _openai_chat_impl(
        organization_id=organization_id,
        agent_id=agent_id,
        run_id=run_id,
        payload=payload,
    )


@app.post("/v1/agents/{agent_id}/runs/{run_id}/openai/responses")
def openai_responses(
    agent_id: str,
    run_id: str,
    payload: dict[str, Any],
    organization_id: str = Depends(require_org),
) -> Any:
    return _openai_responses_impl(
        organization_id=organization_id,
        agent_id=agent_id,
        run_id=run_id,
        payload=payload,
    )


@app.post("/v1/responses")
def openai_compat_responses(
    payload: dict[str, Any],
    x_cynsta_agent_id: Optional[str] = Header(None),
    x_cynsta_run_id: Optional[str] = Header(None),
    organization_id: str = Depends(require_org),
) -> Any:
    agent_id = x_cynsta_agent_id
    if not agent_id:
        raise HTTPException(status_code=400, detail="Missing x-cynsta-agent-id header")
    run_id = x_cynsta_run_id or str(uuid.uuid4())
    return _openai_responses_impl(
        organization_id=organization_id,
        agent_id=agent_id,
        run_id=run_id,
        payload=payload,
    )


def _openai_chat_impl(organization_id: str, agent_id: str, run_id: str, payload: dict[str, Any]) -> Any:
    provider = "openai"
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")
    model = payload.get("model")
    messages = payload.get("messages")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages is required (list)")
    if payload.get("stream", False):
        raise HTTPException(status_code=400, detail="stream=true is not supported in MVP")

    temperature = payload.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise HTTPException(status_code=400, detail="temperature must be a number")
    requested_max = payload.get("max_tokens")
    if requested_max is None:
        requested_max = int(os.getenv("CAP_DEFAULT_MAX_TOKENS", "512"))
    if not isinstance(requested_max, int) or requested_max <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be a positive int")

    inp_rate, out_rate = _model_rates(provider, model)
    tokens_in_est = _estimate_openai_input_tokens(messages)
    cost_in_est = cost_cents(tokens_in_est, inp_rate)

    try:
        budget = store.get_budget(organization_id, agent_id)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    max_affordable = (budget.remaining_cents - cost_in_est) * 1_000_000 // max(1, out_rate)
    if max_affordable <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget for request input")
    max_tokens = min(int(requested_max), int(max_affordable))
    max_tokens = clamp_openai_max_tokens(max_tokens)
    if max_tokens <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget")

    wcec = cost_in_est + cost_cents(max_tokens, out_rate)
    try:
        store.reserve(organization_id, agent_id, run_id, wcec)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    client = OpenAI(api_key=_resolve_provider_key(provider))
    response = None
    completion_text: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached_in: int | None = None
    reasoning_out: int | None = None
    provider_request_id: str | None = None
    termination_reason: str | None = None
    realized = 0
    realized_breakdown: dict[str, Any] | None = None
    error_detail: str | None = None

    try:
        response = call_openai_chat(
            client=client,
            model=model,
            messages=messages,
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=max_tokens,
            stream=False,
        )
        try:
            completion_text = response.choices[0].message.content
        except Exception:
            completion_text = None
        resp_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        provider_request_id = resp_dict.get("id") if isinstance(resp_dict, dict) else None
        usage = resp_dict.get("usage") if isinstance(resp_dict, dict) else None
        if not isinstance(usage, dict):
            tokens_in_raw, tokens_out_raw = extract_openai_usage(response)
            tokens_in = int(tokens_in_raw) if tokens_in_raw is not None else None
            tokens_out = int(tokens_out_raw) if tokens_out_raw is not None else None
        else:
            tokens_in = usage.get("prompt_tokens")
            tokens_out = usage.get("completion_tokens")
            if isinstance(tokens_in, int):
                tokens_in = int(tokens_in)
            else:
                tokens_in = None
            if isinstance(tokens_out, int):
                tokens_out = int(tokens_out)
            else:
                tokens_out = None
            prompt_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                cached = prompt_details.get("cached_tokens")
                cached_in = int(cached) if isinstance(cached, int) else None
            completion_details = usage.get("completion_tokens_details")
            if isinstance(completion_details, dict):
                reason = completion_details.get("reasoning_tokens")
                reasoning_out = int(reason) if isinstance(reason, int) else None

        card = _rate_card(provider, model)
        realized_breakdown = compute_cost_breakdown(
            provider=provider,
            model=model,
            rate_card=card,
            input_tokens=int(tokens_in or tokens_in_est),
            output_tokens=int(tokens_out or 0),
            cached_input_tokens=cached_in,
            reasoning_tokens=reasoning_out,
        )
        realized = int(realized_breakdown["totals"]["realized_cents_ceiled"])
    except Exception as exc:
        termination_reason = "provider_error"
        error_detail = str(exc)
        realized = 0
    finally:
        meta = {
            "agent_id": agent_id,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "wcec_cents": int(wcec),
            "realized_cents": int(realized),
            "tokens_in_est": int(tokens_in_est),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "termination_reason": termination_reason or "ok",
            "max_tokens_clamped": int(max_tokens),
            "timestamp": _utc_now(),
        }
        try:
            store.settle(organization_id, agent_id, run_id, realized, meta)
        except BudgetError as exc:
            # Accounting errors should be loud; do not swallow.
            raise HTTPException(status_code=500, detail=exc.detail) from exc

        store.log_usage(
            {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "wcec_cents": int(wcec),
                "realized_cents": int(realized),
                "termination_reason": termination_reason,
                "meta_json": json.dumps(
                    {
                        "request": {
                            "messages": messages,
                            "model": model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        "usage": getattr(response, "usage", None) if response is not None else None,
                        "provider_request_id": provider_request_id,
                        "billing_breakdown": realized_breakdown,
                        "error": error_detail,
                    },
                    default=str,
                ),
                "created_at": _utc_now(),
            }
        )

        evidence.emit_llm(
            tenant_id=None if organization_id == "local" else organization_id,
            user_id=None,
            provider=provider,
            model=model,
            prompt=messages,
            completion=completion_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            meta=meta,
        )
        evidence.flush()

    if termination_reason:
        raise HTTPException(status_code=502, detail=error_detail or "Provider error")

    return response.model_dump() if hasattr(response, "model_dump") else response


def _count_openai_response_tool_calls(resp_dict: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    output = resp_dict.get("output")
    if not isinstance(output, list):
        return out
    for item in output:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if not isinstance(t, str):
            continue
        # Common pattern: web_search_call, file_search_call, etc.
        if t.endswith("_call"):
            out[t] = out.get(t, 0) + 1
    return out


def _openai_responses_impl(organization_id: str, agent_id: str, run_id: str, payload: dict[str, Any]) -> Any:
    provider = "openai"
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")
    model = payload.get("model")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required")
    if payload.get("stream", False):
        raise HTTPException(status_code=400, detail="stream=true is not supported in MVP")

    requested_max = payload.get("max_output_tokens")
    if requested_max is None:
        requested_max = int(os.getenv("CAP_DEFAULT_MAX_OUTPUT_TOKENS", os.getenv("CAP_DEFAULT_MAX_TOKENS", "512")))
    if not isinstance(requested_max, int) or requested_max <= 0:
        raise HTTPException(status_code=400, detail="max_output_tokens must be a positive int")

    card = _rate_card(provider, model)
    inp_rate, out_rate = card.input_cents_per_1m, card.output_cents_per_1m

    tokens_in_est = _estimate_openai_responses_input_tokens(payload)
    cost_in_est = cost_cents(tokens_in_est, int(inp_rate))

    try:
        budget = store.get_budget(organization_id, agent_id)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    max_affordable = (budget.remaining_cents - cost_in_est) * 1_000_000 // max(1, int(out_rate))
    if max_affordable <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget for request input")
    max_out = min(int(requested_max), int(max_affordable))
    max_out = clamp_openai_max_output_tokens(max_out)
    if max_out <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget")

    wcec = cost_in_est + cost_cents(max_out, int(out_rate))
    try:
        store.reserve(organization_id, agent_id, run_id, wcec)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    client = OpenAI(api_key=_resolve_provider_key(provider))
    response = None
    resp_dict: dict[str, Any] | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached_in: int | None = None
    reasoning_out: int | None = None
    provider_request_id: str | None = None
    tool_calls: dict[str, int] | None = None
    termination_reason: str | None = None
    realized = 0
    realized_breakdown: dict[str, Any] | None = None
    error_detail: str | None = None

    try:
        response = call_openai_responses(client=client, payload=payload, max_output_tokens=max_out)
        resp_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        provider_request_id = resp_dict.get("id") if isinstance(resp_dict, dict) else None
        usage = resp_dict.get("usage") if isinstance(resp_dict, dict) else None
        if isinstance(usage, dict):
            tokens_in = usage.get("input_tokens")
            tokens_out = usage.get("output_tokens")
            inp_details = usage.get("input_tokens_details")
            if isinstance(inp_details, dict):
                cached = inp_details.get("cached_tokens")
                cached_in = int(cached) if isinstance(cached, int) else None
            out_details = usage.get("output_tokens_details")
            if isinstance(out_details, dict):
                reason = out_details.get("reasoning_tokens")
                reasoning_out = int(reason) if isinstance(reason, int) else None
        tool_calls = _count_openai_response_tool_calls(resp_dict) if isinstance(resp_dict, dict) else None

        realized_breakdown = compute_cost_breakdown(
            provider=provider,
            model=model,
            rate_card=card,
            input_tokens=int(tokens_in or tokens_in_est),
            output_tokens=int(tokens_out or 0),
            cached_input_tokens=cached_in,
            reasoning_tokens=reasoning_out,
            tool_calls=tool_calls,
        )
        realized = int(realized_breakdown["totals"]["realized_cents_ceiled"])
    except Exception as exc:
        termination_reason = "provider_error"
        error_detail = str(exc)
        realized = 0
    finally:
        meta = {
            "agent_id": agent_id,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "wcec_cents": int(wcec),
            "realized_cents": int(realized),
            "tokens_in_est": int(tokens_in_est),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "termination_reason": termination_reason or "ok",
            "max_tokens_clamped": int(max_out),
            "timestamp": _utc_now(),
        }
        try:
            store.settle(organization_id, agent_id, run_id, realized, meta)
        except BudgetError as exc:
            raise HTTPException(status_code=500, detail=exc.detail) from exc

        store.log_usage(
            {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "wcec_cents": int(wcec),
                "realized_cents": int(realized),
                "termination_reason": termination_reason,
                "meta_json": json.dumps(
                    {
                        "request": payload,
                        "response": resp_dict,
                        "provider_request_id": provider_request_id,
                        "tool_calls": tool_calls,
                        "billing_breakdown": realized_breakdown,
                        "error": error_detail,
                    },
                    default=str,
                ),
                "created_at": _utc_now(),
            }
        )

        evidence.emit_llm(
            tenant_id=None if organization_id == "local" else organization_id,
            user_id=None,
            provider=provider,
            model=model,
            prompt=payload.get("input"),
            completion=None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            meta=meta,
        )
        evidence.flush()

    if termination_reason:
        raise HTTPException(status_code=502, detail=error_detail or "Provider error")

    return resp_dict or response


@app.post("/v1/agents/{agent_id}/runs/{run_id}/gemini/generateContent")
def gemini_generate_content(
    agent_id: str,
    run_id: str,
    payload: dict[str, Any],
    organization_id: str = Depends(require_org),
) -> dict[str, Any]:
    provider = "gemini"
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required")
    if not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="prompt is required (string)")
    temperature = payload.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise HTTPException(status_code=400, detail="temperature must be a number")
    requested_max = payload.get("max_tokens")
    if requested_max is None:
        requested_max = int(os.getenv("CAP_DEFAULT_MAX_TOKENS", "512"))
    if not isinstance(requested_max, int) or requested_max <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be a positive int")

    inp_rate, out_rate = _model_rates(provider, model)
    tokens_in_est = estimate_tokens_text(prompt)
    cost_in_est = cost_cents(tokens_in_est, inp_rate)

    try:
        budget = store.get_budget(organization_id, agent_id)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    max_affordable = (budget.remaining_cents - cost_in_est) * 1_000_000 // max(1, out_rate)
    if max_affordable <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget for request input")
    max_tokens = min(int(requested_max), int(max_affordable))
    if max_tokens <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget")

    wcec = cost_in_est + cost_cents(max_tokens, out_rate)
    try:
        store.reserve(organization_id, agent_id, run_id, wcec)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    completion_text: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    grounding_queries: int | None = None
    termination_reason: str | None = None
    realized = 0
    realized_breakdown: dict[str, Any] | None = None
    error_detail: str | None = None
    raw: dict[str, Any] | None = None

    try:
        raw = call_gemini_generate_content(
            api_key=_resolve_provider_key(provider),
            model=model,
            prompt=prompt,
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=max_tokens,
        )
        completion_text = extract_gemini_completion(raw)
        tokens_in_raw, tokens_out_raw = extract_gemini_usage(raw)
        tokens_in = int(tokens_in_raw) if tokens_in_raw is not None else None
        tokens_out = int(tokens_out_raw) if tokens_out_raw is not None else None
        grounding = raw.get("groundingMetadata") if isinstance(raw, dict) else None
        if isinstance(grounding, dict):
            queries = grounding.get("webSearchQueries")
            if isinstance(queries, list):
                grounding_queries = len(queries)

        card = _rate_card(provider, model)
        realized_breakdown = compute_cost_breakdown(
            provider=provider,
            model=model,
            rate_card=card,
            input_tokens=int(tokens_in or tokens_in_est),
            output_tokens=int(tokens_out or 0),
            grounding_queries=grounding_queries,
        )
        realized = int(realized_breakdown["totals"]["realized_cents_ceiled"])
    except Exception as exc:
        termination_reason = "provider_error"
        error_detail = str(exc)
        realized = 0
    finally:
        meta = {
            "agent_id": agent_id,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "wcec_cents": int(wcec),
            "realized_cents": int(realized),
            "tokens_in_est": int(tokens_in_est),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "termination_reason": termination_reason or "ok",
            "max_tokens_clamped": int(max_tokens),
            "timestamp": _utc_now(),
        }
        try:
            store.settle(organization_id, agent_id, run_id, realized, meta)
        except BudgetError as exc:
            raise HTTPException(status_code=500, detail=exc.detail) from exc

        store.log_usage(
            {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "wcec_cents": int(wcec),
                "realized_cents": int(realized),
                "termination_reason": termination_reason,
                "meta_json": json.dumps(
                    {"request": payload, "response": raw, "billing_breakdown": realized_breakdown, "error": error_detail},
                    default=str,
                ),
                "created_at": _utc_now(),
            }
        )

        evidence.emit_llm(
            tenant_id=None if organization_id == "local" else organization_id,
            user_id=None,
            provider=provider,
            model=model,
            prompt=prompt,
            completion=completion_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            meta=meta,
        )
        evidence.flush()

    if termination_reason:
        raise HTTPException(status_code=502, detail=error_detail or "Provider error")

    return {"completion": completion_text, "usage": {"tokens_in": tokens_in, "tokens_out": tokens_out}}


@app.post("/v1/agents/{agent_id}/runs/{run_id}/anthropic/messages")
def anthropic_messages(
    agent_id: str,
    run_id: str,
    payload: dict[str, Any],
    organization_id: str = Depends(require_org),
) -> dict[str, Any]:
    provider = "anthropic"
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")
    model = payload.get("model")
    messages = payload.get("messages")
    system = payload.get("system")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages is required (list)")
    if system is not None and not isinstance(system, str):
        raise HTTPException(status_code=400, detail="system must be a string")
    temperature = payload.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise HTTPException(status_code=400, detail="temperature must be a number")
    requested_max = payload.get("max_tokens")
    if requested_max is None:
        requested_max = int(os.getenv("CAP_DEFAULT_MAX_TOKENS", "512"))
    if not isinstance(requested_max, int) or requested_max <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be a positive int")
    if payload.get("stream", False):
        raise HTTPException(status_code=400, detail="stream=true is not supported in MVP")

    tokens_in_est = _estimate_anthropic_input_tokens(system, messages)
    card = _anthropic_rate_card_for_accounting(_rate_card(provider, model))
    inp_rate, out_rate, _, _ = apply_context_cliff_to_rates(card, tokens_in_est)
    cost_in_est = cost_cents(tokens_in_est, int(inp_rate))

    try:
        budget = store.get_budget(organization_id, agent_id)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    max_affordable = (budget.remaining_cents - cost_in_est) * 1_000_000 // max(1, int(out_rate))
    if max_affordable <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget for request input")
    max_tokens = min(int(requested_max), int(max_affordable))
    if max_tokens <= 0:
        raise HTTPException(status_code=402, detail="Insufficient budget")

    wcec = cost_in_est + cost_cents(max_tokens, int(out_rate))
    try:
        store.reserve(organization_id, agent_id, run_id, wcec)
    except BudgetError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    completion_text: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    termination_reason: str | None = None
    realized = 0
    realized_breakdown: dict[str, Any] | None = None
    error_detail: str | None = None
    raw: dict[str, Any] | None = None

    try:
        raw = call_anthropic_messages(
            api_key=_resolve_provider_key(provider),
            model=model,
            system=system,
            messages=messages,
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=max_tokens,
        )
        completion_text = extract_anthropic_completion(raw)
        tokens_in_raw, tokens_out_raw = extract_anthropic_usage(raw)
        tokens_in = int(tokens_in_raw) if tokens_in_raw is not None else None
        tokens_out = int(tokens_out_raw) if tokens_out_raw is not None else None
        usage = raw.get("usage") if isinstance(raw, dict) else None
        cache_write = None
        cache_read = None
        if isinstance(usage, dict):
            cw = usage.get("cache_creation_input_tokens")
            cr = usage.get("cache_read_input_tokens")
            cache_write = int(cw) if isinstance(cw, int) else None
            cache_read = int(cr) if isinstance(cr, int) else None

        card = _anthropic_rate_card_for_accounting(_rate_card(provider, model))
        realized_breakdown = compute_cost_breakdown(
            provider=provider,
            model=model,
            rate_card=card,
            input_tokens=int(tokens_in or tokens_in_est),
            output_tokens=int(tokens_out or 0),
            cache_write_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        )
        realized = int(realized_breakdown["totals"]["realized_cents_ceiled"])
    except Exception as exc:
        termination_reason = "provider_error"
        error_detail = str(exc)
        realized = 0
    finally:
        meta = {
            "agent_id": agent_id,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "wcec_cents": int(wcec),
            "realized_cents": int(realized),
            "tokens_in_est": int(tokens_in_est),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "termination_reason": termination_reason or "ok",
            "max_tokens_clamped": int(max_tokens),
            "timestamp": _utc_now(),
        }
        try:
            store.settle(organization_id, agent_id, run_id, realized, meta)
        except BudgetError as exc:
            raise HTTPException(status_code=500, detail=exc.detail) from exc

        store.log_usage(
            {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "wcec_cents": int(wcec),
                "realized_cents": int(realized),
                "termination_reason": termination_reason,
                "meta_json": json.dumps(
                    {"request": payload, "response": raw, "billing_breakdown": realized_breakdown, "error": error_detail},
                    default=str,
                ),
                "created_at": _utc_now(),
            }
        )

        evidence.emit_llm(
            tenant_id=None if organization_id == "local" else organization_id,
            user_id=None,
            provider=provider,
            model=model,
            prompt={"system": system, "messages": messages},
            completion=completion_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            meta=meta,
        )
        evidence.flush()

    if termination_reason:
        raise HTTPException(status_code=502, detail=error_detail or "Provider error")

    return {
        "completion": completion_text,
        "raw": raw,
        "usage": {"tokens_in": tokens_in, "tokens_out": tokens_out},
        "billing_breakdown": realized_breakdown,
    }
