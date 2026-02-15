from __future__ import annotations

import sys
from pathlib import Path

# Local dev fallback: allow importing sibling spendguard-engine without install.
_engine_src = Path(__file__).resolve().parents[2] / "spendguard-engine" / "src"
if _engine_src.exists():
    engine_src_str = str(_engine_src)
    if engine_src_str not in sys.path:
        sys.path.insert(0, engine_src_str)
