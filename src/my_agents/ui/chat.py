"""Chat from the dashboard: runs the same LangGraph graph as `my-agents chat`.

Each message runs in a background thread so the page can poll for progress.
When a tool needs approval, the run pauses (LangGraph interrupt) and the page
shows the request; your decision resumes it. Conversations are checkpointed in
the same SQLite file as the CLI, so `my-agents chat --thread <id>` can resume them.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path

from ..agents import load_agents, load_router_config
from ..graph import build_graph
from ..store import Store, now
from ..tracing import Tracer


class ChatManager:
    def __init__(self, store: Store, tracer: Tracer):
        from langgraph.checkpoint.sqlite import SqliteSaver

        self.store = store
        self.tracer = tracer
        self.agents = load_agents()
        db = Path(store.path).with_name("checkpoints.sqlite")
        self.graph = build_graph(self.agents, load_router_config(), tracer, store=store,
                                 checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
        self.status: dict[str, dict] = {}  # thread_id -> {state, request_id, approval, error}
        self.lock = threading.Lock()

    # ---- threads --------------------------------------------------------------------

    def threads(self) -> list[dict]:
        rows = self.store.db.execute("SELECT * FROM chat_threads ORDER BY updated DESC LIMIT 100").fetchall()
        return [dict(r) | {"state": self.status.get(r["id"], {}).get("state", "idle")} for r in rows]

    def new_thread(self) -> str:
        return uuid.uuid4().hex[:8]

    def _touch(self, thread_id: str, text: str) -> None:
        self.store.db.execute(
            "INSERT INTO chat_threads (id, title, created, updated) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET updated = excluded.updated",
            [thread_id, " ".join(text.split())[:80], now(), now()])
        self.store.db.commit()

    def state(self, thread_id: str) -> dict:
        values = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values or {}
        st = self.status.get(thread_id, {"state": "idle"})
        return {"thread_id": thread_id, "messages": values.get("messages", []), **st}

    # ---- running --------------------------------------------------------------------

    def send(self, thread_id: str, text: str, agent: str | None) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("empty message")
        if len(text) > 20_000:
            raise ValueError("message too long (20,000 characters max)")
        if agent and agent not in self.agents:
            raise ValueError(f"unknown agent {agent!r}")
        if not thread_id.replace("-", "").isalnum() or len(thread_id) > 64:
            raise ValueError("invalid thread id")
        with self.lock:
            if self.status.get(thread_id, {}).get("state") in ("running", "approval"):
                raise ValueError("this conversation is still working; wait for it to finish")
            request_id = uuid.uuid4().hex
            self.status[thread_id] = {"state": "running", "request_id": request_id}
        self._touch(thread_id, text)
        payload = {"messages": [{"role": "user", "content": text}], "forced_agent": agent or None}
        threading.Thread(target=self._run, args=(thread_id, payload, request_id), daemon=True).start()
        return {"request_id": request_id}

    def decide(self, thread_id: str, decision: str) -> dict:
        from langgraph.types import Command

        if decision not in ("yes", "no", "all"):
            raise ValueError("decision must be yes, no or all")
        with self.lock:
            st = self.status.get(thread_id) or {}
            if st.get("state") != "approval":
                raise ValueError("nothing is waiting for approval in this conversation")
            if decision == "all" and not st["approval"].get("allow_all"):
                raise ValueError("this action can only be approved one at a time")
            self.status[thread_id] = {"state": "running", "request_id": st["request_id"]}
        threading.Thread(target=self._run, args=(thread_id, Command(resume=decision), st["request_id"]),
                         daemon=True).start()
        return {"ok": True}

    def _run(self, thread_id: str, payload, request_id: str) -> None:
        config = {"configurable": {"thread_id": thread_id, "request_id": request_id}}
        try:
            approval = None
            for chunk in self.graph.stream(payload, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    approval = chunk["__interrupt__"][0].value
            new = ({"state": "approval", "request_id": request_id, "approval": approval} if approval
                   else {"state": "idle", "request_id": request_id})
        except Exception as e:  # shown in the chat instead of crashing the server
            new = {"state": "error", "request_id": request_id, "error": f"{type(e).__name__}: {e}"}
        with self.lock:
            self.status[thread_id] = new
        self._touch(thread_id, "")
