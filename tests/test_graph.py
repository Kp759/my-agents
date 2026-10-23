from my_agents.adapters import AdapterError, FakeAdapter
from my_agents.adapters.base import render_transcript
from my_agents.agents import AgentSpec, RouterConfig, load_agents
from my_agents.graph import build_graph, parse_route
from my_agents.tracing import NullTracer

AGENTS = {
    "explainer": AgentSpec("explainer", "explains things", "You explain.", "fake"),
    "examiner": AgentSpec("examiner", "quizzes the user", "You quiz.", "fake"),
}
ROUTER = RouterConfig(backend="fake", model=None, default_agent="explainer")


def make(router_reply="examiner", router_adapter=None):
    tracer = NullTracer()
    adapters = {n: FakeAdapter(reply=f"reply from {n}") for n in AGENTS}
    graph = build_graph(
        AGENTS, ROUTER, tracer, adapters=adapters,
        router_adapter=router_adapter or FakeAdapter(reply=router_reply),
    )
    return graph, adapters, tracer


def ask(graph, text, thread="t1", forced=None):
    return graph.invoke(
        {"messages": [{"role": "user", "content": text}], "forced_agent": forced},
        {"configurable": {"thread_id": thread}},
    )


def test_router_choice_is_followed():
    graph, adapters, tracer = make(router_reply="examiner")
    out = ask(graph, "quiz me on attention")
    assert out["agent"] == "examiner"
    assert out["messages"][-1]["content"] == "reply from examiner"
    assert adapters["explainer"].calls == []
    assert [r["node"] for r in tracer.rows] == ["route", "examiner"]


def test_forced_agent_skips_router():
    router = FakeAdapter(reply="examiner")
    graph, _, _ = make(router_adapter=router)
    out = ask(graph, "anything", forced="explainer")
    assert out["agent"] == "explainer"
    assert router.calls == []


def test_history_is_kept_across_turns_and_agents():
    graph, adapters, _ = make(router_reply="explainer")
    ask(graph, "what is attention?")
    ask(graph, "now quiz me", forced="examiner")
    _, seen = adapters["examiner"].calls[-1]
    assert [m["role"] for m in seen] == ["user", "assistant", "user"]
    assert seen[1]["content"] == "reply from explainer"


def test_threads_are_isolated():
    graph, adapters, _ = make(router_reply="explainer")
    ask(graph, "first", thread="a")
    ask(graph, "second", thread="b")
    _, seen = adapters["explainer"].calls[-1]
    assert len(seen) == 1


def test_router_failure_falls_back_to_default():
    def boom(*_):
        raise AdapterError("router down")

    graph, _, tracer = make(router_adapter=FakeAdapter(reply=boom))
    out = ask(graph, "hello")
    assert out["agent"] == "explainer"
    assert tracer.rows[0]["error"] == "router down"


def test_parse_route():
    names = ["explainer", "examiner"]
    assert parse_route("examiner", names, "explainer") == "examiner"
    assert parse_route("  Examiner.\n", names, "explainer") == "examiner"
    assert parse_route("I'd pick the examiner, not the explainer", names, "explainer") == "examiner"
    assert parse_route("no idea", names, "explainer") == "explainer"


def test_render_transcript():
    assert render_transcript([{"role": "user", "content": "hi"}]) == "hi"
    text = render_transcript([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    assert "User: q1" in text and "Assistant: a1" in text and text.endswith("q2")


def test_shipped_agent_specs_load():
    specs = load_agents()
    assert {"explainer", "examiner"} <= set(specs)
    for spec in specs.values():
        spec.adapter()  # backend name is valid
