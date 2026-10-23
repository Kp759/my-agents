"""Command line entry point.

    my-agents doctor                 check which backends are reachable
    my-agents agents                 list agents, their backends, tools and delegates
    my-agents ask "question"         one request; shows routing, each step and the cost
    my-agents ask -a coder "..."     skip routing and use this agent
    my-agents chat [--thread NAME]   multi-turn chat (saved; resume with the same --thread)
    my-agents trace [RUN_ID]         replay a past request step by step (default: the last one)
    my-agents board [-f]             the shared board; -f follows it live
    my-agents eod [--date D]         end-of-day report of what each agent did
    my-agents ui                     local dashboard: live board, every step, memory, costs
    my-agents tools                  list every tool and its safety flags
    my-agents telegram setup         connect a Telegram bot (one time)
    my-agents telegram               run the Telegram bot: chat and approve from your phone
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import uuid
from datetime import date
from pathlib import Path

import yaml

from .adapters import AdapterError, make_adapter
from .agents import DEFAULT_CONFIG, load_agents, load_router_config
from .board import Board
from .graph import build_graph
from .report import eod_facts, eod_report, format_row, format_totals, totals
from .store import Store
from .tracing import Tracer

CHAT_HELP = "Commands: @<agent> <message> to pick an agent, /agents, /trace, /new, /quit"


def _config() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}


def _pricing() -> dict:
    return _config().get("pricing") or {}


# ---- running requests ------------------------------------------------------------------


class Session:
    def __init__(self, verbose: bool = True):
        from langgraph.checkpoint.sqlite import SqliteSaver

        self.store = Store()
        self.tracer = Tracer()
        self.verbose = verbose
        self.pricing = _pricing()
        db = Path(self.store.path).with_name("checkpoints.sqlite")
        self.graph = build_graph(load_agents(), load_router_config(), self.tracer, store=self.store,
                                 checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))

    def turn(self, thread_id: str, text: str, forced: str | None) -> str:
        from langgraph.types import Command

        run_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id, "request_id": run_id}}
        payload = {"messages": [{"role": "user", "content": text}], "forced_agent": forced}
        shown = 0
        while True:
            interrupted = None
            for chunk in self.graph.stream(payload, config, stream_mode="updates"):
                shown = self._show_new_steps(run_id, shown)
                if "__interrupt__" in chunk:
                    interrupted = chunk["__interrupt__"][0].value
            shown = self._show_new_steps(run_id, shown)
            if interrupted is None:
                break
            payload = Command(resume=self._ask_approval(interrupted))

        state = self.graph.get_state(config).values
        rows = self.tracer.run(run_id)
        if self.verbose:
            print(format_totals(totals(rows, self.pricing)))
        return f"\n[{state['agent']}] {state['messages'][-1]['content']}"

    def _show_new_steps(self, run_id: str, shown: int) -> int:
        if not self.verbose:
            return shown
        rows = self.tracer.run(run_id)
        for row in rows[shown:]:
            print(format_row(row, self.pricing))
        return len(rows)

    @staticmethod
    def _ask_approval(req: dict) -> str:
        print(f"\n⚠  @{req['agent']} wants to run {req['tool']}  ({req.get('reason', 'needs approval')})")
        for k, v in req["args"].items():
            print(f"     {k}: {v}")  # shown in full: you approve exactly what will run
        choices = "[y]es / [N]o / [a]ll writes for this request" if req.get("allow_all") else "[y]es / [N]o"
        try:
            answer = input(f"   Allow? {choices} ").strip().lower()
        except EOFError:
            return "no"
        if answer in ("a", "all") and req.get("allow_all"):
            return "all"
        return "yes" if answer in ("y", "yes") else "no"


# ---- commands --------------------------------------------------------------------------


def cmd_doctor(_args) -> int:
    agents, router = load_agents(), load_router_config()
    needed = {(router.backend, router.model, tuple(router.options.items()))} | {
        (s.backend, s.model, tuple(s.options.items())) for s in agents.values()}
    ok = True
    for backend, model, opts in sorted(needed, key=str):
        up = make_adapter(backend, model, **dict(opts)).available()
        ok &= up
        print(f"  {'✓' if up else '✗'} {backend:<10} {model or '(default)'}")
    if not ok:
        print("\nInstall or start the missing backends, or change them in agents/*.yaml and config.yaml.")
    from .accounts import describe

    print("\naccounts (pinned in config.yaml → accounts):")
    for line in describe():
        print(f"  {line}")
    return 0 if ok else 1


def cmd_agents(_args) -> int:
    router = load_router_config()
    print(f"router: {router.backend} {router.model or ''} · entry: {router.entry} · default: @{router.default_agent}\n")
    for spec in load_agents().values():
        print(f"  @{spec.name:<18} {spec.backend} {spec.model or ''}")
        print(f"      {spec.description}")
        if spec.tools:
            print(f"      tools: {', '.join(spec.tools)}")
        if spec.delegates:
            print(f"      delegates to: {', '.join('@' + d for d in spec.delegates)}")
    return 0


def _split_mention(line: str) -> tuple[str | None, str]:
    if line.startswith("@"):
        head, _, rest = line.partition(" ")
        return head[1:].lower(), rest.strip()
    return None, line


def cmd_ask(args) -> int:
    session = Session(verbose=not args.quiet)
    try:
        print(session.turn(uuid.uuid4().hex, " ".join(args.text), args.agent))
    except (AdapterError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_chat(args) -> int:
    session = Session(verbose=not args.quiet)
    names = set(load_agents())
    thread_id = args.thread or uuid.uuid4().hex[:8]
    owner = _config().get("owner")
    if owner:
        print(f"{owner}'s agents are ready.")
    print(f"{CHAT_HELP}\nthread: {thread_id}  (resume later with: my-agents chat --thread {thread_id})")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/agents":
            cmd_agents(None)
            continue
        if line == "/trace":
            cmd_trace(argparse.Namespace(run_id=None))
            continue
        if line == "/new":
            thread_id = uuid.uuid4().hex[:8]
            print(f"(new conversation: {thread_id})")
            continue
        forced, text = _split_mention(line)
        if forced and forced not in names:
            print(f"unknown agent @{forced}; try /agents")
            continue
        if not text:
            continue
        try:
            print(session.turn(thread_id, text, forced))
        except (AdapterError, ValueError) as e:
            print(f"error: {e}")


def cmd_trace(args) -> int:
    tracer = Tracer()
    run_id = args.run_id or tracer.last_run_id()
    rows = tracer.run(run_id) if run_id else []
    if not rows:
        print("no trace found")
        return 1
    pricing = _pricing()
    first = next((r for r in rows if r.get("node") == "route"), rows[0])
    print(f"request {run_id}  ({first['ts'][:19]} UTC)\n  “{first.get('input_preview', '')}”\n")
    for row in rows:
        print(format_row(row, pricing))
    print(format_totals(totals(rows, pricing)))
    return 0


def cmd_board(args) -> int:
    board = Board(Store())
    rows = board.recent(limit=args.limit, agent=args.agent)
    for row in rows:
        print(Board.format(row))
    if not args.follow:
        return 0
    last = rows[-1]["id"] if rows else 0
    try:
        while True:
            time.sleep(1)
            for row in board.recent(limit=50, agent=args.agent, after_id=last):
                print(Board.format(row), flush=True)
                last = row["id"]
    except KeyboardInterrupt:
        return 0


def cmd_eod(args) -> int:
    day = date.fromisoformat(args.date) if args.date else date.today()
    cfg = _config()
    store = Store()
    facts = eod_facts(Tracer(), store, day, _pricing())
    report = eod_report(facts, None if args.no_llm else cfg.get("eod"), owner=cfg.get("owner") or "the user")
    out = Path(store.path).parent / "eod" / f"{day.isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"saved: {out}")
    Board(store).post("eod", "report", f"End-of-day report for {day.isoformat()} saved to {out}")
    return 0


def cmd_ui(args) -> int:
    from .ui.server import serve

    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_tools(_args) -> int:
    from .tools import make_toolbox

    for t in make_toolbox(Store(":memory:")).values():
        flags = [f for f, on in (("approval", t.approval), ("writes", t.writes), ("untrusted", t.untrusted)) if on]
        print(f"  {t.name:<20} {', '.join(flags) or '-':<22} {t.description[:80]}")
    print("\n  approval: always asks you · writes: asks you once untrusted content was read · "
          "untrusted: returns outside content")
    return 0


def cmd_telegram(args) -> int:
    from . import telegram

    try:
        telegram.setup() if args.action == "setup" else telegram.run()
    except telegram.TelegramError as e:
        print(f"telegram: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    owner = _config().get("owner") or "My"
    p = argparse.ArgumentParser(prog="my-agents", description=f"{owner}'s agents: a transparent multi-agent team.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check which backends are reachable").set_defaults(fn=cmd_doctor)
    sub.add_parser("agents", help="list agents").set_defaults(fn=cmd_agents)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("-a", "--agent", help="skip routing and use this agent")
    ask.add_argument("-q", "--quiet", action="store_true", help="hide the step-by-step trace")
    ask.add_argument("text", nargs="+")
    ask.set_defaults(fn=cmd_ask)

    chat = sub.add_parser("chat", help="interactive chat")
    chat.add_argument("--thread", help="conversation name; reuse it to resume")
    chat.add_argument("-q", "--quiet", action="store_true", help="hide the step-by-step trace")
    chat.set_defaults(fn=cmd_chat)

    trace = sub.add_parser("trace", help="replay a request step by step")
    trace.add_argument("run_id", nargs="?")
    trace.set_defaults(fn=cmd_trace)

    board = sub.add_parser("board", help="show the shared board")
    board.add_argument("-f", "--follow", action="store_true")
    board.add_argument("-n", "--limit", type=int, default=30)
    board.add_argument("--agent")
    board.set_defaults(fn=cmd_board)

    eod = sub.add_parser("eod", help="end-of-day report")
    eod.add_argument("--date", help="YYYY-MM-DD, default today")
    eod.add_argument("--no-llm", action="store_true", help="numbers only, no written summary")
    eod.set_defaults(fn=cmd_eod)

    ui = sub.add_parser("ui", help="open the local dashboard")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true")
    ui.set_defaults(fn=cmd_ui)
    sub.add_parser("tools", help="list tools and their safety flags").set_defaults(fn=cmd_tools)
    tg = sub.add_parser("telegram", help="chat with your agents from Telegram")
    tg.add_argument("action", nargs="?", choices=["run", "setup"], default="run")
    tg.set_defaults(fn=cmd_telegram)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
