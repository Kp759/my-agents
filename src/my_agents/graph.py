"""The LangGraph graph.

    START → route → <agent> ⇄ tools
                       │  ▲
          delegate ──► │  │ ◄── result returns to the caller
                       ▼  │
                    <other agent> ⇄ tools  …  → END

1. `route` picks the first agent: the local router chooses (entry: router), or every
   message goes to the orchestrator (entry: orchestrator). `@agent` skips both.
2. An agent node calls its model. If the reply is a tool call, `tools` runs it and
   control returns to the same agent; otherwise the reply is final.
3. `delegate` is a tool too: the caller is pushed on a stack, the delegate works on
   the sub-task in isolation, and its final answer comes back to the caller as a
   tool result. Delegation depth and count are capped.
4. Tools marked `approval` pause the graph (`interrupt`) until you say yes or no.
5. Taint: once the current job has read untrusted content (web, files), every tool
   that writes needs approval too, so injected instructions cannot quietly change
   your data or memory. Taint passes to delegates and back to callers.
6. Every step is traced and the graph posts progress to the shared board.

Conversation state is checkpointed per `thread_id`, so chats survive restarts
when a SQLite checkpointer is used.
"""

from __future__ import annotations

import json
import operator
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .adapters import AdapterError, ChatAdapter, Message, make_adapter
from .agents import AgentSpec, RouterConfig
from .board import Board
from .guardrails import wrap_untrusted
from .memory import KINDS, AgentMemory
from .store import Store
from .toolcall import format_result, parse_tool_call, tools_prompt
from .tools import Tool, make_toolbox
from .tracing import Tracer, preview

MAX_TOOL_STEPS = 10  # tool calls per agent per task
MAX_DEPTH = 2  # orchestrator → coder → reviewer, no deeper
MAX_DELEGATIONS = 6  # per user message
DEFAULT_CHECKPOINTER = object()


class State(TypedDict, total=False):
    messages: Annotated[list[Message], operator.add]  # what you and the agents said to each other
    forced_agent: str | None  # `@agent`: skip routing
    agent: str  # agent currently working
    task: str | None  # sub-task text when the current agent was delegated to
    scratch: list[Message]  # the current agent's tool calls and results (not shown to other agents)
    stack: list[dict]  # callers waiting for a delegate's answer
    pending_call: dict | None
    steps: int
    delegations: int
    done: bool
    tainted: bool  # the current job has read untrusted content
    writes_ok: bool  # the user approved all writes for the rest of this request


ROUTER_PROMPT = """You route {owner}'s request (a PhD student) to exactly one agent.

Agents:
{agents}

Reply with only the agent's name, nothing else."""


def parse_route(reply: str, names: list[str], default: str) -> str:
    """Take the first agent name that appears in the router's reply."""
    text = reply.strip().lower().lstrip("@")
    if text in names:
        return text
    hits = [(text.find(n), n) for n in names if n in text]
    return min(hits)[1] if hits else default


def builtin_tools(spec: AgentSpec, agents: dict[str, AgentSpec]) -> list[Tool]:
    """Tools the graph implements itself, because they need the graph's state."""
    out = [
        Tool("post_to_board", "Post a short status update to the shared board all agents and the user can see.",
             {"message": "str"}, fn=None, writes=True),
        Tool("read_board", "Read recent posts on the shared board (what other agents are doing).",
             {"limit": "int, optional", "agent": "str, only this agent's posts, optional"}, fn=None),
        Tool("remember", "Save something to your long-term memory for future sessions: the user's level, goals, "
             "gaps, preferences, or a lesson about what worked. Use kind=profile (always shared) for core facts "
             "about the user that every agent should know.",
             {"content": "str, one self-contained sentence", "kind": f"str, {' | '.join(KINDS)}, optional",
              "shared": "bool, also show to every other agent, optional"}, fn=None, writes=True),
        Tool("forget", "Delete one of your memories (or a shared one) that is wrong or outdated.",
             {"id": "int"}, fn=None, writes=True),
    ]
    if spec.delegates:
        who = "; ".join(f"{n}: {agents[n].description}" for n in spec.delegates)
        out.append(Tool(
            "delegate",
            f"Hand a sub-task to another agent and get its answer back. Available: {who}. "
            "The agent sees only the task text, so make it self-contained (include paths, IDs, context).",
            {"agent": f"str, one of {', '.join(spec.delegates)}", "task": "str"}, fn=None,
        ))
    return out


def build_graph(
    agents: dict[str, AgentSpec],
    router: RouterConfig,
    tracer: Tracer,
    store: Store | None = None,
    adapters: dict[str, ChatAdapter] | None = None,
    router_adapter: ChatAdapter | None = None,
    toolbox: dict[str, Tool] | None = None,
    checkpointer=DEFAULT_CHECKPOINTER,
):
    """`adapters`, `router_adapter` and `toolbox` override the defaults (used by tests).

    Pass `checkpointer=None` when a platform supplies its own (LangGraph Studio).
    """
    names = list(agents)
    if router.default_agent not in agents:
        raise ValueError(f"default agent {router.default_agent!r} is not defined")
    if router.entry == "orchestrator" and router.orchestrator not in agents:
        raise ValueError(f"entry is 'orchestrator' but agent {router.orchestrator!r} is not defined")

    store = store or Store()
    board = Board(store)
    memory = AgentMemory(store)
    mission = f"{router.mission}\n\n" if router.mission else ""
    toolbox = toolbox if toolbox is not None else make_toolbox(store)
    adapters = adapters or {name: spec.adapter() for name, spec in agents.items()}
    router_adapter = router_adapter or make_adapter(router.backend, router.model, **router.options)
    router_system = ROUTER_PROMPT.format(
        owner=router.owner,
        agents="\n".join(f"- {s.name}: {s.description}" for s in agents.values())
    )

    agent_tools: dict[str, dict[str, Tool]] = {}
    for spec in agents.values():
        missing = [t for t in spec.tools if t not in toolbox]
        bad_delegates = [d for d in spec.delegates if d not in agents or d == spec.name]
        if missing or bad_delegates:
            raise ValueError(f"agent {spec.name!r}: unknown tools {missing} / delegates {bad_delegates}")
        agent_tools[spec.name] = {t: toolbox[t] for t in spec.tools} | {
            t.name: t for t in builtin_tools(spec, agents)
        }

    # ---- route ---------------------------------------------------------------------
    def route(state: State, config) -> State:
        fresh = {"forced_agent": None, "task": None, "scratch": [], "stack": [], "pending_call": None,
                 "steps": 0, "delegations": 0, "done": False, "tainted": False, "writes_ok": False}
        ids = _ids(config)
        last = state["messages"][-1]
        forced = state.get("forced_agent")
        if forced:
            if forced not in agents:
                raise ValueError(f"unknown agent {forced!r}; choose from {names}")
            chosen, how = forced, "forced by @mention"
        elif router.entry == "orchestrator":
            chosen, how = router.orchestrator, "entry: orchestrator"
        elif len(names) == 1:
            chosen, how = names[0], "only agent"
        else:
            try:
                completion = router_adapter.complete(router_system, [last])
            except AdapterError as e:
                tracer.record(**ids, node="route", agent=router.default_agent, backend=router_adapter.name,
                              model=router_adapter.model, prompt_chars=len(last["content"]), error=str(e),
                              input_preview=preview(last["content"]), decision="fallback to default")
                return {**fresh, "agent": router.default_agent}
            chosen = parse_route(completion.text, names, router.default_agent)
            tracer.record(**ids, node="route", agent=chosen, backend=router_adapter.name,
                          model=router_adapter.model, prompt_chars=len(last["content"]), completion=completion,
                          input_preview=preview(last["content"]), decision=f"router chose {chosen}",
                          io={"system": router_system, "messages": [last], "output": completion.text})
            return {**fresh, "agent": chosen}
        tracer.record(**ids, node="route", agent=chosen, backend="none", model=None,
                      input_preview=preview(last["content"]), decision=how, latency_ms=0)
        return {**fresh, "agent": chosen}

    # ---- agents --------------------------------------------------------------------
    def make_agent_node(spec: AgentSpec):
        adapter = adapters[spec.name]
        tools_text = tools_prompt(list(agent_tools[spec.name].values()))

        def node(state: State, config) -> State:
            ids = _ids(config)
            task, scratch, steps = state.get("task"), state.get("scratch") or [], state.get("steps", 0)
            inputs = ([{"role": "user", "content": task}] if task else state["messages"]) + scratch
            if not scratch:  # first step on this job
                board.post(spec.name, "start", task or state["messages"][-1]["content"], ids["thread_id"])

            # Prompt = shared mission + the agent's role + its memory (+ board) + tool instructions.
            system = mission + spec.system_prompt.rstrip() + memory.for_prompt(spec.name)
            if spec.board_context:
                recent = board.recent(limit=8)
                if recent:
                    system += "\n\n## Recent board posts\n" + wrap_untrusted(
                        "\n".join(Board.format(r) for r in recent))
            system += tools_text

            try:
                completion = adapter.complete(system, inputs)
            except AdapterError as e:
                tracer.record(**ids, node=spec.name, agent=spec.name, backend=adapter.name, model=adapter.model,
                              prompt_chars=_chars(system, inputs), error=str(e), depth=len(state.get("stack") or []),
                              io={"system": system, "messages": inputs, "output": None})
                board.post(spec.name, "error", str(e), ids["thread_id"])
                raise

            call = parse_tool_call(completion.text)
            tracer.record(**ids, node=spec.name, agent=spec.name, backend=adapter.name, model=adapter.model,
                          prompt_chars=_chars(system, inputs), completion=completion,
                          depth=len(state.get("stack") or []), step=steps,
                          action=f"call {call['tool']}" if call else "final answer",
                          io={"system": system, "messages": inputs, "output": completion.text})

            if call and steps < MAX_TOOL_STEPS:
                return {"scratch": scratch + [{"role": "assistant", "content": completion.text}],
                        "pending_call": call, "steps": steps + 1}

            final = completion.text
            if call:  # over budget: stop instead of looping forever
                final = f"(stopped after {MAX_TOOL_STEPS} tool calls) " + final
            board.post(spec.name, "done", final, ids["thread_id"])

            stack = state.get("stack") or []
            if stack:  # return the answer to whoever delegated
                caller = stack[-1]
                tracer.record(**ids, node="return", agent=spec.name, backend="graph", model=None, latency_ms=0,
                              decision=f"{spec.name} → {caller['agent']}", output_preview=preview(final))
                tainted = bool(state.get("tainted"))
                return {
                    "agent": caller["agent"],
                    "task": caller["task"],
                    "steps": caller["steps"],
                    "stack": stack[:-1],
                    "pending_call": None,
                    # a delegate that read untrusted content may pass injected text back up
                    "tainted": caller["tainted"] or tainted,
                    "scratch": caller["scratch"] + [
                        {"role": "user", "content": format_result(f"delegate to {spec.name}", final, untrusted=tainted)}
                    ],
                }
            return {"messages": [{"role": "assistant", "content": final, "agent": spec.name}], "scratch": [],
                    "pending_call": None, "done": True}

        return node

    # ---- tools ---------------------------------------------------------------------
    def tools_node(state: State, config) -> State:
        agent = state["agent"]
        call = state["pending_call"]
        name, args = call["tool"], call["args"]
        tool = agent_tools[agent].get(name)
        scratch = state.get("scratch") or []

        # Ask first. Nothing above this line has side effects, because the node
        # re-runs from the top when the graph resumes after the interrupt.
        updates: State = {}
        taint_gate = bool(tool and tool.writes and state.get("tainted") and not state.get("writes_ok"))
        if tool and (tool.approval or taint_gate):
            reason = ("this action always needs approval" if tool.approval else
                      "this job read untrusted content (web or files), so writes need approval")
            answer = interrupt({"agent": agent, "tool": name, "args": args, "reason": reason,
                                "allow_all": not tool.approval})
            decision = _decision(answer)
            tracer.record(**_ids(config), node="approval", agent=agent, backend="user", model=None, latency_ms=0,
                          tool=name, decision=decision, args=preview(json.dumps(args, default=str)), reason=reason)
            if decision == "no":
                return _tool_result(state, name, "The user declined this action. Ask what they want instead.")
            if decision == "all" and not tool.approval:
                updates["writes_ok"] = True

        ids = _ids(config)
        if tool is None:
            result, error = None, f"unknown tool {name!r}; available: {sorted(agent_tools[agent])}"
        elif name == "delegate":
            return _delegate(state, args, ids)
        else:
            start = time.monotonic()
            try:
                result, error = _run_builtin(name, args, agent, ids) if tool.fn is None else tool.run(args), None
            except Exception as e:  # tool errors go back to the model, which can retry or explain
                result, error = None, f"{type(e).__name__}: {e}"
            tracer.record(**ids, node="tools", agent=agent, backend="tool", model=None, tool=name,
                          args=preview(json.dumps(args, default=str)), error=error,
                          latency_ms=int((time.monotonic() - start) * 1000),
                          output_preview=preview(json.dumps(result, default=str)) if error is None else None,
                          io={"args": args, "result": result if error is None else error})
            if tool.approval and error is None:
                board.post(agent, "action", f"{name} {json.dumps(args, default=str)}", ids["thread_id"])
            if tool.untrusted and error is None:
                updates["tainted"] = True
        content = (format_result(name, result, untrusted=bool(tool and tool.untrusted)) if error is None
                   else f"[tool error: {name}]\n{error}")
        return {**updates, "scratch": scratch + [{"role": "user", "content": content}], "pending_call": None}

    def _tool_result(state: State, name: str, text: str) -> State:
        return {"scratch": (state.get("scratch") or []) + [{"role": "user", "content": format_result(name, text)}],
                "pending_call": None}

    def _run_builtin(name: str, args: dict, agent: str, ids: dict):
        if name == "post_to_board":
            return board.post(agent, "note", str(args.get("message", "")), ids["thread_id"])
        if name == "read_board":
            return [Board.format(r) for r in board.recent(int(args.get("limit", 20)), args.get("agent"))]
        if name == "remember":
            return memory.remember(agent, str(args.get("content", "")), args.get("kind") or "fact",
                                   bool(args.get("shared")))
        if name == "forget":
            return memory.forget(agent, int(args["id"]))
        raise ValueError(name)

    def _delegate(state: State, args: dict, ids: dict) -> State:
        caller, target, task = state["agent"], str(args.get("agent", "")).lstrip("@"), args.get("task")
        stack = state.get("stack") or []
        problem = (
            "missing 'agent' or 'task'" if not target or not task else
            f"you cannot delegate to {target!r}; allowed: {agents[caller].delegates}"
            if target not in agents[caller].delegates else
            f"delegation depth limit ({MAX_DEPTH}) reached; do it yourself" if len(stack) >= MAX_DEPTH else
            f"delegation limit ({MAX_DELEGATIONS}) for this request reached; answer with what you have"
            if state.get("delegations", 0) >= MAX_DELEGATIONS else None
        )
        tracer.record(**ids, node="delegate", agent=caller, backend="graph", model=None, latency_ms=0,
                      decision=f"{caller} → {target}", args=preview(task), error=problem)
        if problem:
            return _tool_result(state, "delegate", f"Delegation refused: {problem}")
        board.post(caller, "delegate", f"→ @{target}: {task}", ids["thread_id"])
        frame = {"agent": caller, "task": state.get("task"), "scratch": state.get("scratch") or [],
                 "steps": state.get("steps", 0), "tainted": bool(state.get("tainted"))}
        # The delegate inherits taint: a task written after reading untrusted content may carry it.
        return {"stack": stack + [frame], "agent": target, "task": task, "scratch": [], "steps": 0,
                "pending_call": None, "delegations": state.get("delegations", 0) + 1,
                "tainted": bool(state.get("tainted"))}

    # ---- wiring --------------------------------------------------------------------
    def after_agent(state: State) -> str:
        if state.get("pending_call"):
            return "tools"
        if state.get("done"):
            return END
        return state["agent"]  # a delegate finished; back to its caller

    g = StateGraph(State)
    g.add_node("route", route)
    g.add_node("tools", tools_node)
    for spec in agents.values():
        g.add_node(spec.name, make_agent_node(spec))
        g.add_conditional_edges(spec.name, after_agent, {"tools": "tools", END: END, **{n: n for n in names}})
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["agent"], {n: n for n in names})
    g.add_conditional_edges("tools", lambda s: s["agent"], {n: n for n in names})

    if checkpointer is DEFAULT_CHECKPOINTER:
        checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)


def _decision(answer) -> str:
    """'yes', 'no' or 'all' (approve every write for the rest of this request)."""
    if answer is True:
        return "yes"
    text = str(answer).strip().lower()
    if text in ("a", "all"):
        return "all"
    return "yes" if text in ("y", "yes", "approve", "ok") else "no"


def _chars(system: str, messages: list[Message]) -> int:
    return len(system) + sum(len(m["content"]) for m in messages)


def _ids(config) -> dict:
    cfg = (config or {}).get("configurable", {})
    return {"run_id": cfg.get("request_id") or uuid.uuid4().hex, "thread_id": cfg.get("thread_id", "default")}
