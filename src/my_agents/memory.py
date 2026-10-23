"""Each agent's long-term memory, plus a profile every agent shares.

Agents write memories with the `remember` tool. Before every model call, the
agent's own recent memories and the shared profile are added to its system
prompt, so they carry over between sessions. Everything is in the
`agent_memory` table and visible (and editable) in the dashboard.
"""

from __future__ import annotations

from .store import Store

SHARED = "*"
KINDS = ("profile", "fact", "preference", "progress", "lesson")
OWN_IN_PROMPT = 20
SHARED_IN_PROMPT = 15


class AgentMemory:
    def __init__(self, store: Store):
        self.store = store

    def remember(self, agent: str, content: str, kind: str = "fact", shared: bool = False) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        owner = SHARED if shared or kind == "profile" else agent
        content = " ".join(content.split())
        dup = self.store.db.execute(
            "SELECT id FROM agent_memory WHERE agent = ? AND lower(content) = lower(?)", [owner, content]).fetchone()
        if dup:
            return {"already_remembered": dup["id"]}
        return self.store.insert("agent_memory", agent=owner, kind=kind, content=content)

    def forget(self, agent: str, id: int) -> str:
        row = self.store.get("agent_memory", id)
        if row["agent"] not in (agent, SHARED):
            raise ValueError(f"memory {id} belongs to @{row['agent']}; you can only forget your own or shared ones")
        self.store.delete("agent_memory", id)
        return f"forgot memory {id}"

    def list(self, agent: str | None = None, limit: int = 200) -> list[dict]:
        return self.store.list("agent_memory", {"agent": agent}, limit=limit)

    def for_prompt(self, agent: str) -> str:
        shared = self.store.list("agent_memory", {"agent": SHARED}, limit=SHARED_IN_PROMPT)
        own = self.store.list("agent_memory", {"agent": agent}, limit=OWN_IN_PROMPT)
        if not shared and not own:
            return ("\n\n## Memory\nYou have no memories yet. Save what you learn about the user "
                    "with `remember`.")
        lines = ["\n\n## Memory (from earlier sessions; ids let you `forget` outdated ones)"]
        if shared:
            lines.append("Shared profile, seen by every agent:")
            lines += [f"- [{m['id']}] ({m['kind']}) {m['content']}" for m in reversed(shared)]
        if own:
            lines.append("Your own memories:")
            lines += [f"- [{m['id']}] ({m['kind']}) {m['content']}" for m in reversed(own)]
        return "\n".join(lines)
