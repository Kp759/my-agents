"""Every model call is written as one JSON line, so cost and routing can be analysed later.

Traces go to `data/traces/YYYY-MM-DD.jsonl` (override with MY_AGENTS_DATA).
`data/` is gitignored: traces contain your conversations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .adapters import Completion

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class Tracer:
    def __init__(self, data_dir: Path | None = None):
        root = data_dir or Path(os.environ.get("MY_AGENTS_DATA", DEFAULT_DATA_DIR))
        self.dir = root / "traces"

    def record(
        self,
        *,
        run_id: str,
        thread_id: str,
        node: str,
        agent: str | None,
        backend: str,
        model: str | None,
        prompt_chars: int,
        completion: Completion | None = None,
        error: str | None = None,
        **extra,
    ) -> dict:
        now = datetime.now(timezone.utc)
        row = {
            "ts": now.isoformat(),
            "run_id": run_id,
            "thread_id": thread_id,
            "node": node,
            "agent": agent,
            "backend": backend,
            "model": model,
            "ok": error is None,
            "error": error,
            "latency_ms": completion.latency_ms if completion else None,
            "prompt_chars": prompt_chars,
            "completion_chars": len(completion.text) if completion else 0,
            "input_tokens": completion.usage.get("input_tokens") if completion else None,
            "output_tokens": completion.usage.get("output_tokens") if completion else None,
            **extra,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{now:%Y-%m-%d}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        return row


class NullTracer(Tracer):
    def __init__(self):
        self.rows: list[dict] = []

    def record(self, **kwargs) -> dict:
        self.rows.append(kwargs)
        return kwargs
