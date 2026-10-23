from .base import AdapterError, ChatAdapter, Completion, Message
from .claude_cli import ClaudeCLIAdapter
from .codex_cli import CodexCLIAdapter
from .fake import FakeAdapter
from .ollama import OllamaAdapter

BACKENDS: dict[str, type] = {
    "claude-cli": ClaudeCLIAdapter,
    "codex-cli": CodexCLIAdapter,
    "ollama": OllamaAdapter,
    "fake": FakeAdapter,
}


def make_adapter(backend: str, model: str | None = None, **options) -> ChatAdapter:
    """`options` are backend-specific keyword arguments, e.g. `think=False` for ollama."""
    try:
        cls = BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}") from None
    if model:
        options["model"] = model
    return cls(**options)


__all__ = [
    "AdapterError", "BACKENDS", "ChatAdapter", "ClaudeCLIAdapter", "CodexCLIAdapter",
    "Completion", "FakeAdapter", "Message", "OllamaAdapter", "make_adapter",
]
