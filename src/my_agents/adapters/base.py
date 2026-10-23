"""The interface every model backend implements.

An adapter turns (system prompt, conversation) into one reply. It hides how the
model is reached: a CLI that uses your subscription, a local model, or a fake
for tests. Agents and the graph only ever see this interface.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Completion:
    text: str
    backend: str
    model: str | None
    latency_ms: int
    usage: dict = field(default_factory=dict)  # token counts, when the backend reports them


class AdapterError(RuntimeError):
    pass


class ChatAdapter(Protocol):
    name: str
    model: str | None

    def complete(self, system: str, messages: list[Message]) -> Completion: ...

    def available(self) -> bool: ...


def render_transcript(messages: list[Message]) -> str:
    """Flatten a conversation into one prompt, for backends that take a single prompt.

    The CLIs run one turn per call, so earlier turns are replayed as text.
    """
    if len(messages) == 1:
        return messages[0]["content"]
    lines = ["Conversation so far:"]
    for m in messages[:-1]:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"\n{speaker}: {m['content']}")
    lines.append(f"\n\nReply to the user's latest message:\n{messages[-1]['content']}")
    return "\n".join(lines)


def on_path(binary: str) -> bool:
    return shutil.which(binary) is not None
