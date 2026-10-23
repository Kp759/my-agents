"""A scripted backend for tests: returns canned replies and records every call."""

from __future__ import annotations

from collections.abc import Callable

from .base import Completion, Message


class FakeAdapter:
    name = "fake"

    def __init__(self, reply: str | Callable[[str, list[Message]], str] = "ok", model: str | None = "fake"):
        self.reply = reply
        self.model = model
        self.calls: list[tuple[str, list[Message]]] = []

    def available(self) -> bool:
        return True

    def complete(self, system: str, messages: list[Message]) -> Completion:
        self.calls.append((system, list(messages)))
        text = self.reply(system, messages) if callable(self.reply) else self.reply
        return Completion(text=text, backend=self.name, model=self.model, latency_ms=0)
