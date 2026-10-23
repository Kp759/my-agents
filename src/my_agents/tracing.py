"""Every step of every request is written as one JSON line.

A request (`run_id`) is a sequence of steps: the routing decision, each agent
call, each tool call and each delegation. `my-agents trace` replays them, and
`my-agents eod` summarises a day of them.

Traces go to `data/traces/YYYY-MM-DD.jsonl` (override with MY_AGENTS_DATA). The full
input and output of each step (the exact system prompt, messages, raw reply, tool
arguments and results) go to `data/calls/<run_id>.jsonl`, keyed by `step_id`.
`data/` is gitignored: traces contain your conversations.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .adapters import Completion

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PREVIEW_CHARS = 300


def preview(text: str | None, n: int = PREVIEW_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


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
        prompt_chars: int = 0,
        completion: Completion | None = None,
        error: str | None = None,
        latency_ms: int | None = None,
        io: dict | None = None,
        **extra,
    ) -> dict:
        now = datetime.now(timezone.utc)
        step_id = uuid.uuid4().hex[:12]
        usage = completion.usage if completion else {}
        row = {
            "ts": now.isoformat(),
            "run_id": run_id,
            "step_id": step_id,
            "thread_id": thread_id,
            "node": node,
            "agent": agent,
            "backend": backend,
            "model": completion.model if completion and completion.model else model,
            "ok": error is None,
            "error": error,
            "latency_ms": completion.latency_ms if completion else latency_ms,
            "prompt_chars": prompt_chars,
            "completion_chars": len(completion.text) if completion else 0,
            "input_tokens": usage.get("input_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": usage.get("cost_usd"),
            "output_preview": preview(completion.text) if completion else None,
            **extra,
        }
        self.write(row, now)
        if io is not None:
            self.write_io(run_id, step_id, io)
        return row

    def write_io(self, run_id: str, step_id: str, io: dict) -> None:
        path = self.dir.parent / "calls" / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"step_id": step_id, **io}, default=str) + "\n")

    def load_io(self, run_id: str) -> dict[str, dict]:
        path = self.dir.parent / "calls" / f"{run_id}.jsonl"
        if not path.exists():
            return {}
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {r["step_id"]: r for r in rows}

    def write(self, row: dict, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{now:%Y-%m-%d}.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def load(self, day: str | None = None) -> list[dict]:
        """All rows for one UTC day (YYYY-MM-DD), default today."""
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.dir / f"{day}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def run(self, run_id: str) -> list[dict]:
        """All rows of one request, searching the most recent days first."""
        for path in sorted(self.dir.glob("*.jsonl"), reverse=True)[:7]:
            rows = [r for r in map(json.loads, path.read_text().splitlines()) if r.get("run_id") == run_id]
            if rows:
                return rows
        return []

    def last_run_id(self) -> str | None:
        for path in sorted(self.dir.glob("*.jsonl"), reverse=True)[:7]:
            lines = path.read_text().splitlines()
            if lines:
                return json.loads(lines[-1])["run_id"]
        return None


class NullTracer(Tracer):
    def __init__(self):
        self.rows: list[dict] = []
        self.io: dict[str, dict] = {}

    def write(self, row: dict, now=None) -> None:
        self.rows.append(row)

    def write_io(self, run_id: str, step_id: str, io: dict) -> None:
        self.io[step_id] = io
