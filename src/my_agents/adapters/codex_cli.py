"""ChatGPT models through the official Codex CLI (`codex exec`), using your ChatGPT sign-in.

Codex has no system-prompt flag, so the system prompt is prepended to the input.
It runs read-only in an empty temp directory. `--json` streams events; the last
agent message is the reply and `turn.completed` carries token usage (no price).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time

from ..accounts import codex_env
from .base import AdapterError, Completion, Message, on_path, render_transcript


class CodexCLIAdapter:
    name = "codex-cli"

    def __init__(self, model: str | None = None, timeout: int = 300):
        self.model = model  # None = your Codex default
        self.timeout = timeout

    def available(self) -> bool:
        return on_path("codex")

    def complete(self, system: str, messages: list[Message]) -> Completion:
        prompt = (
            f"<instructions>\n{system}\n</instructions>\n\n"
            "Answer directly in plain text. Do not run commands or edit files.\n\n"
            + render_transcript(messages)
        )
        start = time.monotonic()
        with tempfile.TemporaryDirectory() as cwd:
            cmd = [
                "codex", "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--color", "never",
                "--json",
                # a chat backend needs none of these; each widens what injected text could do
                "--disable", "browser_use", "--disable", "browser_use_external",
                "--disable", "computer_use", "--disable", "apps",
            ]
            if self.model:
                cmd += ["--model", self.model]
            cmd.append("-")  # read the prompt from stdin
            try:
                proc = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True,
                    cwd=cwd, timeout=self.timeout, env=codex_env(),
                )
            except subprocess.TimeoutExpired as e:
                raise AdapterError(f"codex timed out after {self.timeout}s") from e

        text, usage, error = None, {}, None
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                text = item.get("text", "")
            elif event.get("type") == "turn.completed":
                usage = event.get("usage") or {}
            elif event.get("type") in ("error", "turn.failed"):
                error = event.get("message") or (event.get("error") or {}).get("message")
        if proc.returncode != 0 or text is None:
            raise AdapterError(f"codex failed (exit {proc.returncode}): {error or proc.stderr[-500:]}")

        return Completion(
            text=text.strip(),
            backend=self.name,
            model=self.model,
            latency_ms=int((time.monotonic() - start) * 1000),
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0) + usage.get("reasoning_output_tokens", 0),
            },
        )
