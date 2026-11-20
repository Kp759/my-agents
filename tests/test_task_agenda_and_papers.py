import pytest

from my_agents.agents import load_agents
from my_agents.store import Store
from my_agents.tools import make_toolbox


def test_agenda_filters_and_ranks_all_open_tasks_before_limiting():
    store = Store(":memory:")
    for i in range(31):
        store.insert("tasks", title=f"Backlog {i}", status="doing")
    store.insert("tasks", title="Finished", status="done", due="2026-11-18")
    store.insert("tasks", title="Later", due="2026-11-22")
    store.insert("tasks", title="Normal", due="2026-11-20")
    store.insert("tasks", title="Urgent", due="2026-11-20", priority="high")
    store.insert("tasks", title="Overdue", due="2026-11-19")
    store.insert("tasks", title="Legacy date", due="sometime")
    agenda = make_toolbox(store)["task_agenda"]

    def titles(**kwargs):
        return [r["title"] for r in agenda.run(kwargs)]

    assert titles(limit=5) == ["Overdue", "Urgent", "Normal", "Later", "Backlog 0"]
    assert titles(due_through="2026-11-20") == ["Overdue", "Urgent", "Normal"]
    assert titles(due_from="2026-11-20", due_through="2026-11-20") == ["Urgent", "Normal"]
    assert titles(due_from="2026-11-21") == ["Later"]
    assert titles(limit=100)[-1] == "Legacy date"
    specs = load_agents()
    for agent in ("task_runner", "orchestrator", "friday", "phd_tracker"):
        assert agenda.name in specs[agent].tools


@pytest.mark.parametrize("args", [
    {"due_from": "2026-02-29"}, {"due_through": "20261120"},
    {"due_from": "2026-11-21", "due_through": "2026-11-20"},
    {"limit": -1}, {"limit": True}, {"limit": 101},
])
def test_agenda_rejects_invalid_bounds(args):
    with pytest.raises(ValueError):
        make_toolbox(Store(":memory:"))["task_agenda"].run(args)


def test_saved_paper_search_reaches_older_matches_and_combines_filters():
    store = Store(":memory:")
    older = store.insert("papers", title="Sparse attention", authors="Müller", tags="Étude")
    for i in range(501):
        store.insert("papers", title=f"Other {i}")
    recent = store.insert("papers", title="Another sparse model", authors="Müller", tags="Étude", status="read")
    papers = make_toolbox(store)["list_papers"]
    assert papers.run({"tag": "ÉTU", "status": "to_read", "limit": 1}) == [older]
    assert papers.run({"query": "sparse", "tag": "étude"}) == [recent, older]
    assert papers.run({"query": "MÜLLER", "limit": 1}) == [recent]


def test_completing_task_keeps_saved_draft_unless_result_is_supplied():
    box = make_toolbox(Store(":memory:"))
    task = box["add_task"].run({"title": "Draft the abstract"})
    box["update_task"].run({"id": task["id"], "result": "First draft"})
    assert box["complete_task"].run({"id": task["id"]})["result"] == "First draft"
    assert box["complete_task"].run({"id": task["id"], "result": "Final draft"})["result"] == "Final draft"
