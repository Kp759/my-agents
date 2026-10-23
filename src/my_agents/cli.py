"""Command line entry point.

    my-agents doctor                 check which backends are reachable
    my-agents agents                 list agents and their backends
    my-agents ask "question"         one-shot; the router picks the agent
    my-agents ask -a examiner "..."  one-shot with a chosen agent
    my-agents chat                   multi-turn chat; type @agent to switch agents
"""

from __future__ import annotations

import argparse
import sys
import uuid

from .adapters import AdapterError, make_adapter
from .agents import load_agents, load_router_config
from .graph import build_graph
from .tracing import Tracer

CHAT_HELP = """Commands: @<agent> <message> to pick an agent, /agents, /new (fresh conversation), /quit"""


def cmd_doctor(_args) -> int:
    agents, router = load_agents(), load_router_config()
    needed = {(router.backend, router.model)} | {(s.backend, s.model) for s in agents.values()}
    ok = True
    for backend, model in sorted(needed, key=str):
        up = make_adapter(backend, model).available()
        ok &= up
        print(f"  {'✓' if up else '✗'} {backend:<10} {model or '(default)'}")
    if not ok:
        print("\nInstall or start the missing backends, or change them in agents/*.yaml and config.yaml.")
    return 0 if ok else 1


def cmd_agents(_args) -> int:
    router = load_router_config()
    print(f"router: {router.backend} {router.model or ''} (default agent: {router.default_agent})\n")
    for spec in load_agents().values():
        print(f"  @{spec.name:<12} {spec.backend} {spec.model or ''}\n      {spec.description}")
    return 0


def _graph():
    return build_graph(load_agents(), load_router_config(), Tracer())


def _turn(graph, thread_id: str, text: str, forced: str | None) -> tuple[str, str]:
    config = {"configurable": {"thread_id": thread_id, "run_id": uuid.uuid4().hex}}
    out = graph.invoke({"messages": [{"role": "user", "content": text}], "forced_agent": forced}, config)
    return out["agent"], out["messages"][-1]["content"]


def _split_mention(line: str) -> tuple[str | None, str]:
    if line.startswith("@"):
        head, _, rest = line.partition(" ")
        return head[1:].lower(), rest.strip()
    return None, line


def cmd_ask(args) -> int:
    graph = _graph()
    try:
        agent, reply = _turn(graph, uuid.uuid4().hex, " ".join(args.text), args.agent)
    except (AdapterError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"[{agent}]\n{reply}")
    return 0


def cmd_chat(_args) -> int:
    graph = _graph()
    names = set(load_agents())
    thread_id = uuid.uuid4().hex
    print(CHAT_HELP)
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
        if line == "/new":
            thread_id = uuid.uuid4().hex
            print("(new conversation)")
            continue

        forced, text = _split_mention(line)
        if forced and forced not in names:
            print(f"unknown agent @{forced}; try /agents")
            continue
        if not text:
            continue
        try:
            agent, reply = _turn(graph, thread_id, text, forced)
        except AdapterError as e:
            print(f"error: {e}")
            continue
        print(f"\n[{agent}] {reply}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="my-agents", description="Personal PhD agents on any model.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check which backends are reachable").set_defaults(fn=cmd_doctor)
    sub.add_parser("agents", help="list agents").set_defaults(fn=cmd_agents)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("-a", "--agent", help="skip the router and use this agent")
    ask.add_argument("text", nargs="+")
    ask.set_defaults(fn=cmd_ask)
    sub.add_parser("chat", help="interactive chat").set_defaults(fn=cmd_chat)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
