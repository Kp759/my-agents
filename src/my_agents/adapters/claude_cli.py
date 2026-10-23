"""Claude through the official Claude Code CLI (`claude -p`), using your Claude subscription.

Tools, MCP servers and session files are switched off, so the CLI behaves like a
plain chat model. It runs in an empty temp directory so no project CLAUDE.md leaks in.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time

from .base import AdapterError, Completion, Message, on_path, render_transcript


class ClaudeCLIAdapter:
    name = "claude-cli"

    def __init__(self, model: str | None = "sonnet", timeout: int = 300):
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return on_path("claude")

    def complete(self, system: str, messages: list[Message]) -> Completion:
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt", system,
        ]
        if self.model:
            cmd += ["--model", self.model]

        start = time.monotonic()
        with tempfile.TemporaryDirectory() as cwd:
            try:
                proc = subprocess.run(
                    cmd, input=render_transcript(messages), capture_output=True,
                    text=True, cwd=cwd, timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise AdapterError(f"claude timed out after {self.timeout}s") from e
        latency_ms = int((time.monotonic() - start) * 1000)

        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise AdapterError(f"claude returned non-JSON (exit {proc.returncode}): {proc.stderr[-500:]}") from e
        if out.get("is_error"):
            raise AdapterError(f"claude error: {out.get('result') or out.get('subtype')}")

        usage = out.get("usage") or {}
        models = list((out.get("modelUsage") or {}).keys())
        return Completion(
            text=out.get("result", ""),
            backend=self.name,
            model=models[0] if models else self.model,
            latency_ms=latency_ms,
            usage={
                "input_tokens": usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0),
                "cached_input_tokens": usage.get("cache_read_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                # What the call would cost at API prices; on a subscription you are not billed this.
                "cost_usd": out.get("total_cost_usd"),
            },
        )
