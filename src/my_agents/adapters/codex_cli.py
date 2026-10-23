"""ChatGPT models through the official Codex CLI (`codex exec`), using your ChatGPT sign-in.

Codex has no system-prompt flag, so the system prompt is prepended to the input.
It runs read-only in an empty temp directory, and the final message is read from
the file passed to `-o`.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

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
            out_file = Path(cwd) / "last_message.txt"
            cmd = [
                "codex", "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--color", "never",
                "-o", str(out_file),
            ]
            if self.model:
                cmd += ["--model", self.model]
            cmd.append("-")  # read the prompt from stdin
            try:
                proc = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True,
                    cwd=cwd, timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise AdapterError(f"codex timed out after {self.timeout}s") from e
            if proc.returncode != 0 or not out_file.exists():
                raise AdapterError(f"codex failed (exit {proc.returncode}): {proc.stderr[-500:]}")
            text = out_file.read_text().strip()

        return Completion(
            text=text,
            backend=self.name,
            model=self.model,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
