"""Dashboard chat: messages run in the background, approvals pause and resume."""

import json
import time

import pytest

from my_agents.adapters import FakeAdapter
from my_agents.agents import AgentSpec, RouterConfig
from my_agents.graph import build_graph
from my_agents.store import Store
from my_agents.tracing import NullTracer
from my_agents.chat import ChatManager


def call(tool, **args):
    return "```json\n" + json.dumps({"tool": tool, "args": args}) + "\n```"


def manager(*replies):
    queue = list(replies)
    specs = {"task_runner": AgentSpec("task_runner", "tasks", "x", "fake", tools=["add_task", "delete_task"])}
    store = Store(":memory:")
    m = ChatManager.__new__(ChatManager)  # skip loading the real agents and checkpointer
    m.store, m.tracer, m.agents, m.status = store, NullTracer(), specs, {}
    import threading
    m.lock = threading.Lock()
    m.graph = build_graph(specs, RouterConfig("fake", None, "task_runner"), m.tracer, store=store,
                          adapters={"task_runner": FakeAdapter(reply=lambda *_: queue.pop(0))},
                          router_adapter=FakeAdapter())
    return m, store


def wait(m, thread, until=("idle", "approval", "error"), timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        st = m.state(thread)
        if st["state"] in until:
            return st
        time.sleep(0.02)
    pytest.fail(f"still {m.state(thread)['state']}")


def test_send_runs_in_background_and_labels_agent():
    m, _ = manager("Hello Dr.K")
    m.send("t1", "hi", None)
    st = wait(m, "t1")
    assert st["state"] == "idle"
    assert st["messages"][-1] == {"role": "assistant", "content": "Hello Dr.K", "agent": "task_runner"}
    assert m.threads()[0]["title"] == "hi"


def test_approval_pauses_and_resumes_from_the_page():
    m, store = manager(call("delete_task", id=1), "Deleted it.")
    store.insert("tasks", title="old")
    m.send("t1", "delete task 1", None)
    st = wait(m, "t1")
    assert st["state"] == "approval" and st["approval"]["tool"] == "delete_task"
    with pytest.raises(ValueError):
        m.send("t1", "another message", None)  # busy while waiting
    with pytest.raises(ValueError):
        m.decide("t1", "all")  # always-approve tools can't be bulk-approved
    m.decide("t1", "yes")
    st = wait(m, "t1", until=("idle", "error"))
    assert st["messages"][-1]["content"] == "Deleted it." and store.list("tasks") == []


def test_deny_from_the_page():
    m, store = manager(call("delete_task", id=1), "OK, kept it.")
    store.insert("tasks", title="old")
    m.send("t1", "delete task 1", None)
    wait(m, "t1")
    m.decide("t1", "no")
    wait(m, "t1", until=("idle", "error"))
    assert len(store.list("tasks")) == 1


@pytest.mark.parametrize("thread,text,agent", [("t1", "", None), ("t1", "hi", "nobody"), ("../x", "hi", None)])
def test_bad_chat_input_is_rejected(thread, text, agent):
    m, _ = manager()
    with pytest.raises(ValueError):
        m.send(thread, text, agent)


def test_post_requires_custom_header(tmp_path, monkeypatch):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    monkeypatch.setenv("MY_AGENTS_DATA", str(tmp_path))
    from my_agents.ui.server import API, make_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(API()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/api/chat/send"
    try:
        # a plain cross-site form POST cannot set custom headers
        req = urllib.request.Request(url, data=b'{"thread_id":"t","text":"delete everything"}', method="POST",
                                     headers={"Content-Type": "text/plain"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
    finally:
        server.shutdown()
