"""The LangGraph graph: a router node picks an agent, then that agent's node replies.

    START → route ─┬─→ explainer ─→ END
                   ├─→ examiner  ─→ END
                   └─→ ...       ─→ END

Conversation state is kept per `thread_id` by the checkpointer, so a chat can
switch agents mid-conversation while every agent sees the full history.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .adapters import AdapterError, ChatAdapter, Message, make_adapter
from .agents import AgentSpec, RouterConfig
from .tracing import Tracer


class State(TypedDict, total=False):
    messages: Annotated[list[Message], operator.add]
    forced_agent: str | None  # set by the user (e.g. `@examiner`); skips the router
    agent: str  # the agent chosen for this turn


ROUTER_PROMPT = """You route a PhD student's request to exactly one agent.

Agents:
{agents}

Reply with only the agent's name, nothing else."""


def parse_route(reply: str, names: list[str], default: str) -> str:
    """Take the first agent name that appears in the router's reply."""
    text = reply.strip().lower()
    if text in names:
        return text
    hits = [(text.find(n), n) for n in names if n in text]
    return min(hits)[1] if hits else default


def build_graph(
    agents: dict[str, AgentSpec],
    router: RouterConfig,
    tracer: Tracer,
    adapters: dict[str, ChatAdapter] | None = None,
    router_adapter: ChatAdapter | None = None,
    checkpointer=None,
):
    """`adapters` and `router_adapter` override the specs' backends (used by tests)."""
    if router.default_agent not in agents:
        raise ValueError(f"default agent {router.default_agent!r} is not defined")
    adapters = adapters or {name: spec.adapter() for name, spec in agents.items()}
    router_adapter = router_adapter or make_adapter(router.backend, router.model)
    names = list(agents)
    router_system = ROUTER_PROMPT.format(
        agents="\n".join(f"- {s.name}: {s.description}" for s in agents.values())
    )

    def route(state: State, config) -> State:
        forced = state.get("forced_agent")
        if forced:
            if forced not in agents:
                raise ValueError(f"unknown agent {forced!r}; choose from {names}")
            return {"agent": forced, "forced_agent": None}
        if len(names) == 1:
            return {"agent": names[0]}

        last = state["messages"][-1]
        ids = _ids(config)
        try:
            completion = router_adapter.complete(router_system, [last])
        except AdapterError as e:
            tracer.record(**ids, node="route", agent=None, backend=router_adapter.name,
                          model=router_adapter.model, prompt_chars=len(last["content"]), error=str(e))
            return {"agent": router.default_agent}
        chosen = parse_route(completion.text, names, router.default_agent)
        tracer.record(**ids, node="route", agent=chosen, backend=router_adapter.name,
                      model=router_adapter.model, prompt_chars=len(last["content"]),
                      completion=completion, router_reply=completion.text.strip()[:200])
        return {"agent": chosen}

    def make_agent_node(spec: AgentSpec):
        adapter = adapters[spec.name]

        def node(state: State, config) -> State:
            messages = state["messages"]
            prompt_chars = sum(len(m["content"]) for m in messages)
            try:
                completion = adapter.complete(spec.system_prompt, messages)
            except AdapterError as e:
                tracer.record(**_ids(config), node=spec.name, agent=spec.name, backend=adapter.name,
                              model=adapter.model, prompt_chars=prompt_chars, error=str(e))
                raise
            tracer.record(**_ids(config), node=spec.name, agent=spec.name, backend=adapter.name,
                          model=adapter.model, prompt_chars=prompt_chars, completion=completion)
            return {"messages": [{"role": "assistant", "content": completion.text}]}

        return node

    g = StateGraph(State)
    g.add_node("route", route)
    for spec in agents.values():
        g.add_node(spec.name, make_agent_node(spec))
        g.add_edge(spec.name, END)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["agent"], {n: n for n in names})
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def _ids(config) -> dict:
    cfg = (config or {}).get("configurable", {})
    return {"run_id": cfg.get("run_id") or uuid.uuid4().hex, "thread_id": cfg.get("thread_id", "default")}
