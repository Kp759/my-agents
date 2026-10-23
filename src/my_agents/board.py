"""The shared board: a feed of what every agent is working on.

The graph posts automatically (start, delegate, action, done); agents can also
post and read it through tools. You can watch it with `my-agents board -f`.
"""

from __future__ import annotations

from .store import Store


class Board:
    def __init__(self, store: Store):
        self.store = store

    def post(self, agent: str, kind: str, content: str, thread_id: str | None = None) -> dict:
        content = " ".join(content.split())
        if len(content) > 300:
            content = content[:299] + "…"
        return self.store.insert("board", agent=agent, kind=kind, content=content, thread_id=thread_id)

    def recent(self, limit: int = 20, agent: str | None = None, after_id: int = 0) -> list[dict]:
        rows = self.store.list("board", {"agent": agent}, limit=max(1, min(int(limit), 200)))
        return [r for r in reversed(rows) if r["id"] > after_id]

    @staticmethod
    def format(row: dict) -> str:
        return f"[{row['created'][11:19]}] @{row['agent']:<18} {row['kind']:<8} {row['content']}"
