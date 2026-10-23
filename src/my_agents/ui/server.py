"""A local dashboard over the traces, board, memory and store. Standard library only.

    my-agents ui            → http://127.0.0.1:8765

It only binds to 127.0.0.1 (your machine). It reads the same files the agents
write, so it shows requests live while you use `my-agents chat` in a terminal.

Hardening: requests must carry a Host of 127.0.0.1 or localhost (stops DNS
rebinding, where a website re-points its domain at your machine). POSTs must
carry the `X-My-Agents` header, which a web page on another site cannot send
without a CORS preflight this server never approves, and POSTs from another
origin are refused. Responses forbid framing, sniffing and outside scripts.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from ..agents import DEFAULT_CONFIG, load_agents, load_router_config
from ..board import Board
from ..memory import AgentMemory
from ..report import eod_facts, eod_report, row_cost, totals
from ..store import Store
from ..tracing import Tracer
from .chat import ChatManager

INDEX = Path(__file__).with_name("index.html")
TABLES = ("tasks", "experiments", "milestones", "papers", "skills", "quiz_results", "notes")


class API:
    def __init__(self):
        self.store = Store()
        self.tracer = Tracer()
        self.board = Board(self.store)
        self.memory = AgentMemory(self.store)
        self._chat = None

    @property
    def chat(self) -> ChatManager:
        if self._chat is None:  # built on first use: loads agents and the checkpointer
            self._chat = ChatManager(self.store, self.tracer)
        return self._chat

    def _config(self) -> dict:
        return yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}

    def _pricing(self) -> dict:
        return self._config().get("pricing") or {}

    def runs(self, days: int = 3) -> list[dict]:
        rows = []
        today = datetime.now(timezone.utc).date()
        for d in range(days):
            rows += self.tracer.load((today - timedelta(days=d)).isoformat())
        by_run = defaultdict(list)
        for r in rows:
            by_run[r["run_id"]].append(r)
        out = []
        for run_id, rs in by_run.items():
            rs.sort(key=lambda r: r["ts"])
            route = next((r for r in rs if r["node"] == "route"), None)
            final = next((r for r in reversed(rs) if r.get("action") == "final answer" and not r.get("depth")), None)
            path = []
            for r in rs:
                if r["node"] not in ("route", "tools", "delegate", "return") and (not path or path[-1] != r["agent"]):
                    path.append(r["agent"])
            t = totals(rs, self._pricing())
            out.append({
                "run_id": run_id, "ts": rs[0]["ts"], "thread_id": rs[0].get("thread_id"),
                "input": (route or {}).get("input_preview", ""), "first_agent": (route or {}).get("agent"),
                "path": path, "final_agent": (final or {}).get("agent"), "answer": (final or {}).get("output_preview"),
                "steps": len(rs), **t,
            })
        out.sort(key=lambda r: r["ts"], reverse=True)
        return out

    def run(self, run_id: str) -> dict:
        rows = self.tracer.run(run_id)
        pricing = self._pricing()
        for r in rows:
            r["cost_estimate"] = row_cost(r, pricing)
        return {"rows": rows, "io": self.tracer.load_io(run_id), "totals": totals(rows, pricing) if rows else None}

    def agents(self) -> dict:
        router = load_router_config()
        specs = load_agents()
        today = self.tracer.load()
        stats = defaultdict(lambda: {"calls": 0, "tool_calls": 0, "cost_usd": 0.0, "tokens": 0})
        for r in today:
            s = stats[r.get("agent")]
            if r["node"] == "tools":
                s["tool_calls"] += 1
            elif r["node"] not in ("route", "delegate", "return"):
                s["calls"] += 1
                s["cost_usd"] += row_cost(r, self._pricing()) or 0
                s["tokens"] += (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
        mem_counts = defaultdict(int)
        for m in self.memory.list(limit=10_000):
            mem_counts[m["agent"]] += 1
        return {
            "router": {"entry": router.entry, "backend": router.backend, "model": router.model,
                       "default_agent": router.default_agent, "orchestrator": router.orchestrator},
            "mission": router.mission,
            "owner": router.owner,
            "agents": [{
                "name": s.name, "description": s.description, "backend": s.backend, "model": s.model,
                "tools": s.tools, "delegates": s.delegates, "board_context": s.board_context,
                "system_prompt": s.system_prompt, "memories": mem_counts.get(s.name, 0),
                "today": stats.get(s.name, {"calls": 0, "tool_calls": 0, "cost_usd": 0.0, "tokens": 0}),
            } for s in specs.values()],
            "shared_memories": mem_counts.get("*", 0),
        }

    def memory_view(self) -> dict:
        return {
            "memories": self.memory.list(limit=1000),
            "tables": {t: self.store.list(t, limit=200) for t in TABLES},
        }

    def eod(self, day: str | None, generate: bool, llm: bool) -> dict:
        d = date.fromisoformat(day) if day else date.today()
        path = Path(self.store.path).parent / "eod" / f"{d.isoformat()}.md"
        facts = eod_facts(self.tracer, self.store, d, self._pricing())
        if generate:
            report = eod_report(facts, self._config().get("eod") if llm else None,
                                owner=self._config().get("owner") or "the user")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report)
        return {"day": d.isoformat(), "facts": facts, "report": path.read_text() if path.exists() else None}


def make_handler(api: API):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body, status=200, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                             "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
                             "form-action 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._host_ok():
                return self._send({"error": "forbidden host"}, 403)
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self._send(INDEX.read_bytes(), ctype="text/html; charset=utf-8")
                if url.path == "/api/runs":
                    return self._send(api.runs(int(q.get("days", 3))))
                if m := re.fullmatch(r"/api/run/([\w-]+)", url.path):
                    return self._send(api.run(m.group(1)))
                if url.path == "/api/board":
                    return self._send(api.board.recent(int(q.get("limit", 100)), q.get("agent"),
                                                       int(q.get("after", 0))))
                if url.path == "/api/agents":
                    return self._send(api.agents())
                if url.path == "/api/memory":
                    return self._send(api.memory_view())
                if url.path == "/api/eod":
                    return self._send(api.eod(q.get("date"), False, False))
                if url.path == "/api/chat/threads":
                    return self._send(api.chat.threads())
                if url.path == "/api/chat/new":
                    return self._send({"thread_id": api.chat.new_thread()})
                if url.path == "/api/chat/state":
                    return self._send(api.chat.state(q.get("thread", "")))
                return self._send({"error": "not found"}, 404)
            except Exception as e:
                return self._send({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            if not self._host_ok():
                return self._send({"error": "forbidden host"}, 403)
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            # Only same-origin requests from the dashboard itself may change data.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self._send({"error": "cross-origin request refused"}, 403)
            if self.headers.get("X-My-Agents") != "1":
                return self._send({"error": "missing X-My-Agents header"}, 403)
            length = int(self.headers.get("Content-Length") or 0)
            if length > 100_000:
                return self._send({"error": "request too large"}, 413)
            try:
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            except json.JSONDecodeError:
                return self._send({"error": "invalid JSON"}, 400)
            try:
                if m := re.fullmatch(r"/api/memory/(\d+)/delete", url.path):
                    api.store.delete("agent_memory", int(m.group(1)))
                    return self._send({"ok": True})
                if url.path == "/api/eod":
                    return self._send(api.eod(q.get("date"), True, q.get("llm") == "1"))
                if url.path == "/api/chat/send":
                    return self._send(api.chat.send(str(body.get("thread_id", "")), str(body.get("text", "")),
                                                    body.get("agent") or None))
                if url.path == "/api/chat/decide":
                    return self._send(api.chat.decide(str(body.get("thread_id", "")), str(body.get("decision", ""))))
                return self._send({"error": "not found"}, 404)
            except ValueError as e:
                return self._send({"error": str(e)}, 400)
            except Exception as e:
                return self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(API()))
    url = f"http://127.0.0.1:{port}"
    print(f"dashboard: {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
