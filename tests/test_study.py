"""Study loop: spaced repetition, curriculum, scheduled jobs."""

from datetime import date, datetime

from my_agents import study
from my_agents import telegram as tg
from my_agents.agents import DEFAULT_CONFIG
from my_agents.store import Store
from my_agents.tools import make_toolbox
from test_telegram import FakeClient, bot, msg, wait_for

import yaml


def test_good_answers_space_reviews_out_and_bad_ones_reset():
    ease, interval, reps = 2.5, 0, 0
    intervals = []
    for _ in range(4):
        ease, interval, reps = study.next_schedule(ease, interval, reps, 0.9)
        intervals.append(interval)
    assert intervals[0] == 1 and intervals[1] == 3 and intervals[3] > intervals[2] > 3
    ease2, interval, reps = study.next_schedule(ease, interval, reps, 0.3)
    assert interval == 1 and reps == 0 and ease2 < ease


def test_quiz_results_schedule_the_next_review():
    store = Store(":memory:")
    out = make_toolbox(store)["record_quiz_result"].run({"topic": "Multi-agent debate", "score": 0.4})
    assert out["interval_days"] == 1
    due_tomorrow = study.due(store, today=date.fromisoformat(out["next_review"]))["due"]
    assert [r["topic"] for r in due_tomorrow] == ["Multi-agent debate"]
    study.review(store, "multi-agent DEBATE", 0.9)  # same topic, any case
    assert len(store.list("reviews")) == 1 and store.list("reviews")[0]["reps"] == 1


def test_curriculum_init_is_idempotent_and_staggered():
    store = Store(":memory:")
    first = study.init_curriculum(store)
    assert first["topics_added"] == len(study.load_curriculum()) >= 8 and first["papers_added"] >= 14
    assert study.init_curriculum(store) == {"topics_added": 0, "papers_added": 0}
    assert len(study.due(store)["due"]) == 1  # one new module per day, not all at once


def test_curriculum_links_are_real_sources():
    for m in study.load_curriculum():
        assert m["objectives"] and m["exercise"]
        for r in m["readings"]:
            assert r["url"].startswith(("https://arxiv.org/abs/", "https://www.anthropic.com/", "https://www.wiley.com/"))


def test_jobs_run_once_a_day_on_their_days():
    job = {"name": "career", "at": "18:00", "days": ["sun"]}
    sunday_evening = datetime(2026, 10, 25, 18, 30)
    assert study.is_due(job, sunday_evening, None)
    assert not study.is_due(job, sunday_evening, "2026-10-25")  # already ran today
    assert not study.is_due(job, datetime(2026, 10, 25, 17, 59), None)
    assert not study.is_due(job, datetime(2026, 10, 26, 18, 30), None)  # Monday


def test_shipped_schedule_fills_in_focus_and_owner():
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    jobs = {j["name"]: j for j in study.jobs(cfg)}
    assert {"digest", "quiz", "career"} <= set(jobs) and jobs["quiz"]["sticky"]
    assert "{focus}" not in jobs["digest"]["prompt"] and cfg["owner"] in jobs["digest"]["prompt"]


def test_telegram_runs_due_jobs_and_quiz_replies_stay_with_the_examiner(tmp_path, monkeypatch):
    job = {"name": "quiz", "at": "00:00", "agent": "task_runner", "sticky": True, "prompt": "Quiz me"}
    monkeypatch.setattr(tg, "_full_config", lambda: {"study": {"schedule": [job]}})
    b, client, _ = bot("Question 1?", "Correct. Question 2?", tmp_path=tmp_path)
    b.maybe_run_jobs()
    wait_for(lambda: any("Question 1?" in m["text"] for m in client.sent))
    assert b.state["jobs_ran"]["quiz"] == date.today().isoformat() and b.state["sticky"]["agent"] == "task_runner"
    b.maybe_run_jobs()  # not twice in a day
    wait_for(lambda: b.chat.status[b.thread]["state"] == "idle")
    b.handle(msg("my answer"))
    wait_for(lambda: any("Question 2?" in m["text"] for m in client.sent))
    b.handle(msg("/done"))
    assert "sticky" not in b.state
