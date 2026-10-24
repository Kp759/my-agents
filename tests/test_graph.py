import json

import pytest
from langgraph.types import Command

from my_agents.adapters import AdapterError, FakeAdapter
from my_agents.adapters.base import render_transcript
from my_agents.agents import AgentSpec, RouterConfig, load_agents, load_router_config
from my_agents.board import Board
from my_agents.graph import MAX_TOOL_STEPS, build_graph, parse_route
from my_agents.report import eod_facts, eod_report, format_row, totals
from my_agents.store import Store
from my_agents.toolcall import parse_tool_call
from my_agents.tools import Tool, make_toolbox
from my_agents.tracing import NullTracer


def call(tool, **args):
    return "```json\n" + json.dumps({"tool": tool, "args": args}) + "\n```"


def script(*replies):
    """A fake model that gives these replies in order."""
    queue = list(replies)
    return FakeAdapter(reply=lambda *_: queue.pop(0))


def make(agent_replies, router_reply="explainer", specs=None, toolbox=None, entry="router"):
    specs = specs or {
        "explainer": AgentSpec("explainer", "explains things", "You explain.", "fake"),
        "examiner": AgentSpec("examiner", "quizzes the user", "You quiz.", "fake"),
    }
    store = Store(":memory:")
    tracer = NullTracer()
    adapters = {n: (agent_replies[n] if n in agent_replies else FakeAdapter(reply=f"reply from {n}")) for n in specs}
    router = RouterConfig("fake", None, default_agent=next(iter(specs)), entry=entry,
                          orchestrator="orchestrator")
    graph = build_graph(specs, router, tracer, store=store, adapters=adapters,
                        router_adapter=FakeAdapter(reply=router_reply), toolbox=toolbox)
    return graph, adapters, tracer, store


def ask(graph, text, thread="t1", forced=None):
    return graph.invoke({"messages": [{"role": "user", "content": text}], "forced_agent": forced},
                        {"configurable": {"thread_id": thread, "request_id": f"r-{thread}-{text[:10]}"}})


# ---- routing -----------------------------------------------------------------------


def test_router_choice_is_followed():
    graph, adapters, tracer, _ = make({}, router_reply="examiner")
    out = ask(graph, "quiz me on attention")
    assert out["agent"] == "examiner"
    assert out["messages"][-1]["content"] == "reply from examiner"
    assert adapters["explainer"].calls == []
    assert [r["node"] for r in tracer.rows] == ["route", "examiner"]


def test_forced_agent_skips_router():
    graph, _, tracer, _ = make({}, router_reply="examiner")
    out = ask(graph, "anything", forced="explainer")
    assert out["agent"] == "explainer"
    assert tracer.rows[0]["decision"] == "forced by @mention"


def test_history_is_kept_across_turns_and_agents():
    graph, adapters, _, _ = make({})
    ask(graph, "what is attention?")
    ask(graph, "now quiz me", forced="examiner")
    _, seen = adapters["examiner"].calls[-1]
    assert [m["role"] for m in seen] == ["user", "assistant", "user"]
    assert seen[1]["content"] == "reply from explainer"


def test_threads_are_isolated():
    graph, adapters, _, _ = make({})
    ask(graph, "first", thread="a")
    ask(graph, "second", thread="b")
    assert len(adapters["explainer"].calls[-1][1]) == 1


def test_router_failure_falls_back_to_default():
    graph, _, tracer, _ = make({})
    graph2 = build_graph(
        {"explainer": AgentSpec("explainer", "x", "x", "fake"), "examiner": AgentSpec("examiner", "y", "y", "fake")},
        RouterConfig("fake", None, "explainer"), tracer, store=Store(":memory:"),
        adapters={"explainer": FakeAdapter(), "examiner": FakeAdapter()},
        router_adapter=FakeAdapter(reply=lambda *_: (_ for _ in ()).throw(AdapterError("router down"))),
    )
    out = ask(graph2, "hello")
    assert out["agent"] == "explainer"
    assert tracer.rows[0]["error"] == "router down"


def test_parse_route():
    names = ["explainer", "examiner"]
    assert parse_route("examiner", names, "explainer") == "examiner"
    assert parse_route("  @Examiner.\n", names, "explainer") == "examiner"
    assert parse_route("I'd pick the examiner, not the explainer", names, "explainer") == "examiner"
    assert parse_route("no idea", names, "explainer") == "explainer"


# ---- tools ---------------------------------------------------------------------------


def test_tool_loop_runs_tool_then_answers():
    specs = {"task_runner": AgentSpec("task_runner", "tasks", "You run tasks.", "fake",
                                      tools=["add_task", "list_tasks"])}
    model = script(call("add_task", title="write intro"), "Added task 1.")
    graph, _, tracer, store = make({"task_runner": model}, specs=specs)
    out = ask(graph, "add a task to write the intro")
    assert out["messages"][-1]["content"] == "Added task 1."
    assert store.list("tasks")[0]["title"] == "write intro"
    # the second model call saw its own tool call and the result
    _, seen = model.calls[1]
    assert seen[-2]["role"] == "assistant" and "add_task" in seen[-2]["content"]
    assert seen[-1]["content"].startswith("[tool result: add_task]")
    assert [r["node"] for r in tracer.rows] == ["route", "task_runner", "tools", "task_runner"]
    # full inputs and outputs are captured for every step
    assert any("args" in io for io in tracer.io.values())
    assert any(io.get("system", "").startswith("You run tasks.") for io in tracer.io.values())


def test_tool_errors_go_back_to_the_model():
    specs = {"t": AgentSpec("t", "tasks", "x", "fake", tools=["update_task"])}
    model = script(call("update_task", id=99, status="done"), "That task does not exist.")
    graph, _, _, _ = make({"t": model}, specs=specs)
    out = ask(graph, "finish task 99")
    assert "[tool error: update_task]" in model.calls[1][1][-1]["content"]
    assert out["messages"][-1]["content"] == "That task does not exist."


def test_tool_budget_stops_infinite_loops():
    specs = {"t": AgentSpec("t", "tasks", "x", "fake", tools=["list_tasks"])}
    graph, _, _, _ = make({"t": FakeAdapter(reply=call("list_tasks"))}, specs=specs)
    out = ask(graph, "loop forever")
    assert out["messages"][-1]["content"].startswith(f"(stopped after {MAX_TOOL_STEPS} tool calls)")


def test_unlisted_tool_is_refused():
    specs = {"t": AgentSpec("t", "tasks", "x", "fake", tools=["list_tasks"])}
    model = script(call("delete_task", id=1), "ok")
    graph, _, _, _ = make({"t": model}, specs=specs)
    ask(graph, "delete")
    assert "unknown tool 'delete_task'" in model.calls[1][1][-1]["content"]


# ---- approval ------------------------------------------------------------------------


def approval_setup():
    specs = {"t": AgentSpec("t", "tasks", "x", "fake", tools=["add_task", "delete_task"])}
    model = script(call("delete_task", id=1), "Deleted.")
    graph, _, tracer, store = make({"t": model}, specs=specs)
    store.insert("tasks", title="old task")
    return graph, model, store


def test_approval_pauses_then_runs_on_yes():
    graph, _, store = approval_setup()
    out = ask(graph, "delete task 1")
    req = out["__interrupt__"][0].value
    assert (req["agent"], req["tool"], req["args"], req["allow_all"]) == ("t", "delete_task", {"id": 1}, False)
    assert len(store.list("tasks")) == 1  # nothing happened yet
    out = graph.invoke(Command(resume=True), {"configurable": {"thread_id": "t1", "request_id": "r"}})
    assert store.list("tasks") == []
    assert out["messages"][-1]["content"] == "Deleted."


def test_approval_declined_does_nothing():
    graph, model, store = approval_setup()
    ask(graph, "delete task 1")
    graph.invoke(Command(resume=False), {"configurable": {"thread_id": "t1", "request_id": "r"}})
    assert len(store.list("tasks")) == 1
    assert "declined" in model.calls[1][1][-1]["content"]


# ---- delegation ----------------------------------------------------------------------


def delegation_specs():
    return {
        "orchestrator": AgentSpec("orchestrator", "lead", "You lead.", "fake", delegates=["coder", "reviewer"]),
        "coder": AgentSpec("coder", "codes", "You code.", "fake", delegates=["reviewer"]),
        "reviewer": AgentSpec("reviewer", "reviews", "You review.", "fake"),
    }


def test_delegation_returns_result_to_caller():
    orch = script(call("delegate", agent="coder", task="write fizzbuzz"), "Coder wrote it and reviewer approved.")
    coder = script(call("delegate", agent="reviewer", task="review fizzbuzz"), "fizzbuzz done; review: APPROVE")
    reviewer = script("APPROVE")
    graph, _, tracer, store = make({"orchestrator": orch, "coder": coder, "reviewer": reviewer},
                                   specs=delegation_specs(), entry="orchestrator")
    out = ask(graph, "build fizzbuzz")
    assert out["agent"] == "orchestrator"
    assert out["messages"][-1]["content"] == "Coder wrote it and reviewer approved."
    assert reviewer.calls[0][1] == [{"role": "user", "content": "review fizzbuzz"}]  # isolated context
    assert "fizzbuzz done" in orch.calls[1][1][-1]["content"]  # result came back as a tool result
    assert len(out["messages"]) == 2  # sub-agent chatter is not added to the user conversation
    kinds = [(b["agent"], b["kind"]) for b in Board(store).recent(50)]
    assert ("orchestrator", "delegate") in kinds and ("reviewer", "done") in kinds
    assert [r["node"] for r in tracer.rows if r["node"] in ("delegate", "return")] == [
        "delegate", "delegate", "return", "return"]


def test_delegation_depth_is_capped():
    specs = delegation_specs()
    specs["reviewer"].delegates = ["coder"]
    orch = script(call("delegate", agent="coder", task="a"), "done")
    coder = script(call("delegate", agent="reviewer", task="b"), "coder done")
    reviewer = script(call("delegate", agent="coder", task="c"), "reviewer done")
    graph, _, _, _ = make({"orchestrator": orch, "coder": coder, "reviewer": reviewer},
                          specs=specs, entry="orchestrator")
    ask(graph, "go")
    assert "depth limit" in reviewer.calls[1][1][-1]["content"]


def test_cannot_delegate_to_unlisted_agent():
    specs = delegation_specs()
    orch = script(call("delegate", agent="reviewer", task="x"), "ok")
    specs["orchestrator"].delegates = ["coder"]
    graph, _, _, _ = make({"orchestrator": orch}, specs=specs, entry="orchestrator")
    ask(graph, "go")
    assert "cannot delegate" in orch.calls[1][1][-1]["content"]


# ---- reporting -----------------------------------------------------------------------


def test_trace_totals_and_eod(tmp_path):
    orch = script(call("delegate", agent="coder", task="x"), "all done")
    graph, _, tracer, store = make({"orchestrator": orch}, specs=delegation_specs(), entry="orchestrator")
    ask(graph, "go")
    t = totals(tracer.rows)
    assert t["model_calls"] == 3 and t["delegations"] == 1
    assert all(format_row(r) for r in tracer.rows)

    from datetime import date, datetime, timezone

    class DayTracer(NullTracer):
        def load(self, day=None):  # eod reads three UTC days; serve the rows once
            return tracer.rows if day == datetime.now(timezone.utc).date().isoformat() else []

    facts = eod_facts(DayTracer(), store, date.today())
    assert facts["agents"]["orchestrator"]["delegated_to"] == {"coder": 1}
    assert facts["agents"]["coder"]["delegated_from"] == {"orchestrator": 1}
    report = eod_report(facts, writer=None)
    assert "| @coder |" in report


# ---- protocol and specs --------------------------------------------------------------


def test_parse_tool_call():
    assert parse_tool_call(call("list_tasks", status="todo")) == {"tool": "list_tasks", "args": {"status": "todo"}}
    assert parse_tool_call('Let me check.\n```json\n{"tool": "x"}\n```') == {"tool": "x", "args": {}}
    assert parse_tool_call('{"tool": "x", "args": {"a": 1}}') == {"tool": "x", "args": {"a": 1}}
    assert parse_tool_call("Here is the answer: use {curly braces}.") is None
    assert parse_tool_call('```json\n{"not_a_tool": 1}\n```') is None
    # seen from a real model: closing fence without an opening one
    assert parse_tool_call('{"tool": "list_tasks", "args": {"limit": 20}} ```') == {
        "tool": "list_tasks", "args": {"limit": 20}}
    assert parse_tool_call('I will check.\n{"tool": "x", "args": {"q": "a {b}"}}') == {
        "tool": "x", "args": {"q": "a {b}"}}


def test_tool_argument_validation():
    t = Tool("t", "d", {"a": "int", "b": "str, optional"}, fn=lambda a, b=None: a)
    assert t.run({"a": 1}) == 1
    with pytest.raises(ValueError):
        t.run({"b": "x"})
    with pytest.raises(ValueError):
        t.run({"a": 1, "zzz": 2})


def test_render_transcript():
    assert render_transcript([{"role": "user", "content": "hi"}]) == "hi"
    text = render_transcript([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                              {"role": "user", "content": "q2"}])
    assert "User: q1" in text and "Assistant: a1" in text and text.endswith("q2")


def test_shipped_specs_build_a_graph():
    specs = load_agents()
    assert {"orchestrator", "explainer", "examiner", "task_runner", "experiment_tracker", "phd_tracker",
            "lit_surveyor", "trend_scout", "career_coach", "coder", "reviewer", "writer", "friday"} == set(specs)
    graph = build_graph(specs, load_router_config(), NullTracer(), store=Store(":memory:"),
                        adapters={n: FakeAdapter() for n in specs}, router_adapter=FakeAdapter())
    assert {"route", "tools", *specs} <= set(graph.get_graph().nodes)


def test_memory_tools_round_trip():
    tools = make_toolbox(Store(":memory:"))
    exp = tools["log_experiment"].run({"name": "lr sweep", "metrics": {"acc": 0.8}})
    tools["update_experiment"].run({"id": exp["id"], "metrics": {"loss": 0.3}, "status": "done"})
    got = tools["get_experiment"].run({"id": exp["id"]})
    assert got["metrics"] == {"acc": 0.8, "loss": 0.3} and got["status"] == "done"
    tools["set_skill"].run({"skill": "JAX", "level": 2, "target": 4})
    tools["set_skill"].run({"skill": "jax", "level": 3})
    assert [(s["skill"], s["level"]) for s in tools["list_skills"].run({})] == [("JAX", 3)]
    saved = tools["save_paper"].run({"title": "P", "url": "https://x"})
    assert tools["save_paper"].run({"title": "P", "url": "https://x"}) == {"already_saved": saved}


def test_agents_remember_across_sessions_and_share_profile():
    specs = {
        "explainer": AgentSpec("explainer", "explains", "You explain.", "fake"),
        "examiner": AgentSpec("examiner", "quizzes", "You quiz.", "fake"),
    }
    explainer = script(call("remember", content="User struggles with KL divergence", kind="progress"),
                       call("remember", content="User is a PhD student in machine unlearning", kind="profile"),
                       "Noted.", "Welcome back.")
    examiner = script("Question 1?")
    graph, _, _, store = make({"explainer": explainer, "examiner": examiner}, specs=specs)
    ask(graph, "hi", thread="s1")
    ask(graph, "hi again", thread="s2")  # a new session
    system_next_session = explainer.calls[-1][0]
    assert "User struggles with KL divergence" in system_next_session
    assert "machine unlearning" in system_next_session
    ask(graph, "quiz me", thread="s3", forced="examiner")
    examiner_system = examiner.calls[0][0]
    assert "machine unlearning" in examiner_system  # shared profile
    assert "KL divergence" not in examiner_system  # explainer's own memory stays private


def test_mission_is_prepended():
    specs = {"explainer": AgentSpec("explainer", "explains", "You explain.", "fake")}
    model = FakeAdapter()
    graph = build_graph(specs, RouterConfig("fake", None, "explainer", mission="MISSION: get hired"),
                        NullTracer(), store=Store(":memory:"), adapters={"explainer": model},
                        router_adapter=FakeAdapter())
    ask(graph, "hi")
    assert model.calls[0][0].startswith("MISSION: get hired\n\nYou explain.")


def test_owner_name_reaches_router_and_agents(tmp_path):
    from my_agents.agents import load_router_config
    cfg = tmp_path / "config.yaml"
    cfg.write_text("owner: Dr.K\nmission: |\n  You are one of {owner}'s agents.\n"
                   "router: {backend: fake, default_agent: explainer}\n")
    router = load_router_config(cfg)
    assert router.owner == "Dr.K" and router.mission == "You are one of Dr.K's agents."
    specs = {"explainer": AgentSpec("explainer", "x", "You explain.", "fake"),
             "examiner": AgentSpec("examiner", "y", "You quiz.", "fake")}
    route_model, agent_model = FakeAdapter(reply="explainer"), FakeAdapter()
    graph = build_graph(specs, router, NullTracer(), store=Store(":memory:"),
                        adapters={"explainer": agent_model, "examiner": FakeAdapter()}, router_adapter=route_model)
    ask(graph, "hi")
    assert "Dr.K's request" in route_model.calls[0][0]
    assert agent_model.calls[0][0].startswith("You are one of Dr.K's agents.")


def test_claude_native_tool_markup_is_parsed_as_a_call():
    text = ('<invoke name="web_search">\n<parameter name="query">LangGraph release</parameter>\n'
            '<parameter name="limit">8</parameter>\n<parameter name="deep">true</parameter>\n</invoke>')
    assert parse_tool_call(text) == {"tool": "web_search", "args": {"query": "LangGraph release", "limit": 8, "deep": True}}


def test_parameters_with_defaults_are_marked_optional():
    # A parameter that has a default but isn't marked "optional" becomes required, and calls fail.
    for t in make_toolbox(Store(":memory:")).values():
        for name, desc in t.params.items():
            if "default" in desc:
                assert "optional" in desc, f"{t.name}.{name}: {desc!r}"


# ---- search first (grounding) --------------------------------------------------------


def grounded(agent_replies, ground_reply, specs=None, entry="router", search=None):
    """A graph with grounding: auto, a fake web_search, and a router that answers both prompts."""
    from my_agents.graph import GROUND_PROMPT

    searches = []

    def web_search(query, limit=8, recency=None, deep=True):
        searches.append(query)
        if search:
            return search(query)
        return [{"title": "Claude 9 released", "url": "https://example.com/claude9", "snippet": "Out today."}]
    box = {"web_search": Tool("web_search", "search", {"query": "str"}, web_search, untrusted=True),
           "add_task": Tool("add_task", "add", {"title": "str"}, lambda title: {"id": 1}, writes=True)}
    specs = specs or {
        "explainer": AgentSpec("explainer", "explains", "You explain.", "fake", tools=["web_search", "add_task"]),
        "examiner": AgentSpec("examiner", "quizzes", "You quiz.", "fake"),
    }
    router_calls = []

    def route_or_ground(system, messages):
        router_calls.append(system)
        return ground_reply if system == GROUND_PROMPT else "explainer"
    adapters = {n: (agent_replies[n] if n in agent_replies else FakeAdapter(reply=f"reply from {n}")) for n in specs}
    tracer = NullTracer()
    router = RouterConfig("fake", None, default_agent=next(iter(specs)), entry=entry, grounding="auto")
    graph = build_graph(specs, router, tracer, store=Store(":memory:"), adapters=adapters, toolbox=box,
                        router_adapter=FakeAdapter(reply=route_or_ground))
    return graph, adapters, tracer, searches


def test_search_first_puts_results_before_the_agent_answers():
    graph, adapters, tracer, searches = grounded({}, "SEARCH: latest Claude model")
    out = ask(graph, "what's the newest Claude model?")
    assert searches == ["latest Claude model"]
    seen = adapters["explainer"].calls[0][1]
    assert "Searched first" in seen[-1]["content"] and "Claude 9 released" in seen[-1]["content"]
    assert "check the sources against your own knowledge" in seen[-1]["content"]
    assert seen[0]["content"] == "what's the newest Claude model?"  # the question comes first
    assert [r["node"] for r in tracer.rows] == ["route", "ground", "tools", "explainer"]
    assert out["tainted"] and out["sources"]
    assert all(format_row(r) for r in tracer.rows)


def test_no_search_when_the_router_says_no():
    graph, adapters, tracer, searches = grounded({}, "NO")
    ask(graph, "add a task to call the dentist")
    assert searches == [] and len(adapters["explainer"].calls[0][1]) == 1
    assert [r["node"] for r in tracer.rows] == ["route", "ground", "explainer"]


def test_agents_without_web_search_are_not_grounded():
    graph, _, tracer, searches = grounded({}, "SEARCH: x")
    ask(graph, "quiz me", forced="examiner")
    assert searches == [] and "ground" not in [r["node"] for r in tracer.rows]


def test_search_results_stay_in_view_across_tool_steps_and_arm_the_taint_gate():
    agent = script(call("add_task", title="read the Claude 9 post"), "Done.")
    graph, _, _, _ = grounded({"explainer": agent}, "SEARCH: claude 9")
    out = ask(graph, "what's new with Claude?")
    assert "__interrupt__" in out  # a write after reading search results needs approval
    out = graph.invoke(Command(resume="y"), {"configurable": {"thread_id": "t1", "request_id": "r2"}})
    assert out["messages"][-1]["content"] == "Done."
    assert "Searched first" in agent.calls[1][1][1]["content"]  # still there on the second step


def test_a_failed_search_does_not_break_the_answer():
    def boom(_q):
        raise RuntimeError("offline")
    graph, adapters, tracer, searches = grounded({}, "SEARCH: x", search=boom)
    out = ask(graph, "who won the Turing award this year?")
    assert out["messages"][-1]["content"] == "reply from explainer"
    assert len(adapters["explainer"].calls[0][1]) == 1 and not out.get("tainted")
    assert next(r for r in tracer.rows if r["node"] == "tools").get("error")


def test_delegates_are_grounded_on_their_own_task():
    specs = {
        "orchestrator": AgentSpec("orchestrator", "lead", "You lead.", "fake", delegates=["explainer"]),
        "explainer": AgentSpec("explainer", "explains", "You explain.", "fake", tools=["web_search"]),
    }
    orch = script(call("delegate", agent="explainer", task="what is new in LangGraph 1.2?"), "Summary.")
    graph, adapters, _, searches = grounded({"orchestrator": orch}, "SEARCH: LangGraph 1.2 release", specs=specs,
                                            entry="orchestrator")
    ask(graph, "brief me on LangGraph")
    assert searches == ["LangGraph 1.2 release"]  # orchestrator has no web_search; the delegate searched
    assert "Searched first" in adapters["explainer"].calls[0][1][-1]["content"]


def test_parse_ground():
    from my_agents.graph import parse_ground
    assert parse_ground("SEARCH: qwen 3.5 release date") == "qwen 3.5 release date"
    assert parse_ground('**Search:** "best pizza near NJIT"') == "best pizza near NJIT"
    assert parse_ground("NO") is None and parse_ground("hmm, not sure") is None


def test_grounding_is_read_from_config_and_off_by_default():
    assert RouterConfig("fake", None, "x").grounding == "off"
    assert load_router_config().grounding in ("auto", "off")
