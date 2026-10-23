"""Tools over the shared memory store: tasks, experiments, milestones, notes, papers, quizzes, skills."""

from __future__ import annotations

from .. import study
from ..store import Store
from . import Tool

TASK_STATUS = "todo | doing | done"
EXP_STATUS = "planned | running | done | failed"
MILESTONE_STATUS = "planned | in_progress | done | blocked"


def tools(store: Store) -> list[Tool]:
    def add_task(title, details=None, priority=None, due=None):
        return store.insert("tasks", title=title, details=details, priority=priority, due=due)

    def list_tasks(status=None, limit=30):
        return store.list("tasks", {"status": status}, order="status, due IS NULL, due, id", limit=limit)

    def update_task(id, title=None, details=None, status=None, priority=None, due=None, result=None):
        return store.update("tasks", id, title=title, details=details, status=status,
                            priority=priority, due=due, result=result)

    def complete_task(id, result=""):
        return store.update("tasks", id, status="done", result=result)

    def delete_task(id):
        store.delete("tasks", id)
        return f"deleted task {id}"

    def log_experiment(name, hypothesis=None, config=None, status=None, metrics=None, notes=None, next_steps=None):
        return store.insert("experiments", name=name, hypothesis=hypothesis, config=config, status=status,
                            metrics=metrics, notes=notes, next_steps=next_steps)

    def update_experiment(id, status=None, metrics=None, notes=None, next_steps=None, config=None, hypothesis=None):
        if metrics is not None:  # merge, so partial updates don't drop earlier metrics
            metrics = {**store.get("experiments", id)["metrics"], **metrics}
        return store.update("experiments", id, status=status, metrics=metrics, notes=notes,
                            next_steps=next_steps, config=config, hypothesis=hypothesis)

    def list_experiments(status=None, limit=30):
        return store.list("experiments", {"status": status}, limit=limit)

    def get_experiment(id):
        return store.get("experiments", id)

    def add_milestone(title, due=None, status=None, notes=None):
        return store.insert("milestones", title=title, due=due, status=status, notes=notes)

    def update_milestone(id, title=None, due=None, status=None, notes=None):
        return store.update("milestones", id, title=title, due=due, status=status, notes=notes)

    def list_milestones(status=None):
        return store.list("milestones", {"status": status}, order="due IS NULL, due, id", limit=100)

    def add_note(content, kind=None):
        return store.insert("notes", content=content, kind=kind)

    def list_notes(kind=None, limit=20):
        return store.list("notes", {"kind": kind}, limit=limit)

    def save_paper(title, url, authors=None, year=None, summary=None, tags=None, status=None):
        existing = [p for p in store.list("papers", {"url": url}, limit=1)]
        if existing:
            return {"already_saved": existing[0]}
        return store.insert("papers", title=title, url=url, authors=authors, year=year,
                            summary=summary, tags=tags, status=status)

    def update_paper(id, summary=None, tags=None, status=None):
        return store.update("papers", id, summary=summary, tags=tags, status=status)

    def list_papers(status=None, tag=None, limit=30):
        rows = store.list("papers", {"status": status}, limit=500 if tag else limit)
        if tag:
            rows = [r for r in rows if tag.lower() in (r["tags"] or "").lower()][:limit]
        return rows

    def record_quiz_result(topic, score, gaps=None):
        result = store.insert("quiz_results", topic=topic, score=float(score), gaps=gaps)
        r = study.review(store, topic, float(score))  # spaced repetition: when to quiz this again
        return {**result, "next_review": r["due"], "interval_days": r["interval_days"]}

    def list_quiz_results(topic=None, limit=20):
        return store.list("quiz_results", {"topic": topic}, limit=limit)

    def set_skill(skill, level=None, target=None, evidence=None, plan=None):
        return store.upsert_skill(skill, level=level, target=target, evidence=evidence, plan=plan)

    def list_skills():
        return store.list("skills", order="target - level DESC", limit=100)

    return [
        Tool("add_task", "Add a task to the to-do list.",
             {"title": "str", "details": "str, optional", "priority": "str, low|medium|high, optional",
              "due": "str, YYYY-MM-DD, optional"}, add_task, writes=True),
        Tool("list_tasks", "List tasks, open ones first.",
             {"status": f"str, {TASK_STATUS}, optional", "limit": "int, optional"}, list_tasks),
        Tool("update_task", "Edit a task's fields or status.",
             {"id": "int", "title": "str, optional", "details": "str, optional",
              "status": f"str, {TASK_STATUS}, optional", "priority": "str, optional",
              "due": "str, optional", "result": "str, what was produced, optional"}, update_task, writes=True),
        Tool("complete_task", "Mark a task done and record what was produced. Asks the user first.",
             {"id": "int", "result": "str, optional"}, complete_task, approval=True, writes=True),
        Tool("delete_task", "Delete a task permanently. Asks the user first.",
             {"id": "int"}, delete_task, approval=True, writes=True),
        Tool("log_experiment", "Record a new experiment.",
             {"name": "str", "hypothesis": "str, optional", "config": "object, hyperparameters etc., optional",
              "status": f"str, {EXP_STATUS}, optional", "metrics": "object, optional",
              "notes": "str, optional", "next_steps": "str, optional"}, log_experiment, writes=True),
        Tool("update_experiment", "Update an experiment; metrics are merged into existing ones.",
             {"id": "int", "status": f"str, {EXP_STATUS}, optional", "metrics": "object, optional",
              "notes": "str, optional", "next_steps": "str, optional", "config": "object, optional",
              "hypothesis": "str, optional"}, update_experiment, writes=True),
        Tool("list_experiments", "List experiments, newest first.",
             {"status": f"str, {EXP_STATUS}, optional", "limit": "int, optional"}, list_experiments),
        Tool("get_experiment", "Get one experiment in full.", {"id": "int"}, get_experiment),
        Tool("add_milestone", "Add a PhD milestone (qualifier, paper deadline, proposal, defense...).",
             {"title": "str", "due": "str, YYYY-MM-DD, optional",
              "status": f"str, {MILESTONE_STATUS}, optional", "notes": "str, optional"}, add_milestone, writes=True),
        Tool("update_milestone", "Update a milestone.",
             {"id": "int", "title": "str, optional", "due": "str, optional",
              "status": f"str, {MILESTONE_STATUS}, optional", "notes": "str, optional"}, update_milestone, writes=True),
        Tool("list_milestones", "List milestones by due date.",
             {"status": f"str, {MILESTONE_STATUS}, optional"}, list_milestones),
        Tool("add_note", "Save a note (e.g. kind=meeting, idea, feedback).",
             {"content": "str", "kind": "str, optional"}, add_note, writes=True),
        Tool("list_notes", "List recent notes.", {"kind": "str, optional", "limit": "int, optional"}, list_notes),
        Tool("save_paper", "Save a paper to the reading list.",
             {"title": "str", "url": "str", "authors": "str, optional", "year": "int, optional",
              "summary": "str, optional", "tags": "str, comma-separated, optional",
              "status": "str, to_read | read, optional"}, save_paper, writes=True),
        Tool("update_paper", "Update a saved paper's summary, tags or status.",
             {"id": "int", "summary": "str, optional", "tags": "str, optional",
              "status": "str, to_read | read, optional"}, update_paper, writes=True),
        Tool("list_papers", "List saved papers.",
             {"status": "str, optional", "tag": "str, optional", "limit": "int, optional"}, list_papers),
        Tool("record_quiz_result", "Record how the user did on a quiz topic; also schedules its next review "
             "(spaced repetition). Use the topic name exactly as due_reviews shows it.",
             {"topic": "str", "score": "float, 0 to 1", "gaps": "str, what they missed, optional"},
             record_quiz_result, writes=True),
        Tool("list_quiz_results", "List past quiz results to find weak topics.",
             {"topic": "str, optional", "limit": "int, optional"}, list_quiz_results),
        Tool("due_reviews", "Topics due for review today (spaced repetition), weakest first, plus upcoming ones.",
             {"limit": "int, optional"}, lambda limit=10: study.due(store, limit=int(limit))),
        Tool("add_review_topic", "Schedule a topic for spaced-repetition quizzes (e.g. after reading a paper).",
             {"topic": "str", "notes": "str, what to know about it, optional"},
             lambda topic, notes=None: study.add_review(store, topic, notes), writes=True),
        Tool("set_skill", "Create or update a skill in the user's skill profile.",
             {"skill": "str", "level": "int, current level 1-5, optional", "target": "int, needed level 1-5, optional",
              "evidence": "str, optional", "plan": "str, optional"}, set_skill, writes=True),
        Tool("list_skills", "List the user's skills, biggest gaps first.", {}, list_skills),
    ]
