"""Local models through Ollama's HTTP API. Free, private, no account needed."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import AdapterError, Completion, Message


class OllamaAdapter:
    name = "ollama"

    def __init__(
        self,
        model: str | None = "qwen3:8b",
        timeout: int = 300,
        host: str | None = None,
        think: bool | None = None,
        keep_alive: str = "30m",
    ):
        self.model = model
        self.think = think  # reasoning models (e.g. Qwen3): False skips the thinking phase; None = model default
        self.timeout = timeout
        self.keep_alive = keep_alive  # how long Ollama keeps the model in memory after a call
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def complete(self, system: str, messages: list[Message]) -> Completion:
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if self.think is not None:
            body["think"] = self.think
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            raise AdapterError(f"ollama request failed: {e}") from e

        return Completion(
            text=out["message"]["content"],
            backend=self.name,
            model=self.model,
            latency_ms=int((time.monotonic() - start) * 1000),
            usage={
                "input_tokens": out.get("prompt_eval_count", 0),
                "output_tokens": out.get("eval_count", 0),
                "cost_usd": 0.0,
            },
        )
