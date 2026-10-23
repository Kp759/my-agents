"""Calls the real backends. Uses a little of your subscription quota.

    pytest -m live
"""

import pytest

from my_agents.adapters import ClaudeCLIAdapter, CodexCLIAdapter, OllamaAdapter

BACKENDS = [ClaudeCLIAdapter(model="haiku"), CodexCLIAdapter(), OllamaAdapter(think=False)]


@pytest.mark.live
@pytest.mark.parametrize("adapter", BACKENDS, ids=lambda a: a.name)
def test_backend_answers_and_keeps_history(adapter):
    if not adapter.available():
        pytest.skip(f"{adapter.name} not available")
    out = adapter.complete(
        "Answer with a single word.",
        [
            {"role": "user", "content": "My favourite colour is teal. Remember it."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What is my favourite colour?"},
        ],
    )
    assert "teal" in out.text.lower()
    assert out.latency_ms > 0
