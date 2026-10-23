"""The study loop: spaced repetition, a curriculum, and scheduled study jobs.

Spaced repetition (SM-2, simplified): every quiz score reschedules its topic. Well-known
topics come back after longer and longer gaps (1 day, 3 days, then interval × ease);
a weak score (< 0.6) brings the topic back tomorrow and makes it come back sooner from
then on. The Examiner records scores with `record_quiz_result`; `due_reviews` tells it
what to ask today.

Scheduled jobs (config.yaml → study.schedule) are prompts sent to an agent at a set
time, for example the morning paper digest or the evening quiz. The Telegram bot runs
them; `my-agents study run JOB` runs one now in the terminal.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

from .store import Store

CURRICULUM = Path(__file__).resolve().parents[2] / "study" / "mas_curriculum.yaml"
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
PASS = 0.6  # below this a review counts as forgotten


# ---- spaced repetition ------------------------------------------------------------------


def next_schedule(ease: float, interval: int, reps: int, score: float) -> tuple[float, int, int]:
    """(ease, interval_days, reps) after a review with `score` in [0, 1]."""
    score = max(0.0, min(1.0, float(score)))
    if score < PASS:
        return max(1.3, ease - 0.2), 1, 0
    ease = max(1.3, min(3.0, ease + (score - 0.8) * 0.5))  # 1.0 → +0.1, 0.6 → -0.1
    interval = 1 if reps == 0 else 3 if reps == 1 else max(interval + 1, round(interval * ease))
    return ease, interval, reps + 1


def find_review(store: Store, topic: str) -> dict | None:
    row = store.db.execute("SELECT id FROM reviews WHERE lower(topic) = lower(?)", [topic.strip()]).fetchone()
    return store.get("reviews", row["id"]) if row else None


def add_review(store: Store, topic: str, notes: str | None = None, due: str | None = None) -> dict:
    existing = find_review(store, topic)
    if existing:
        return {"already_scheduled": existing}
    return store.insert("reviews", topic=topic.strip(), notes=notes, due=due or date.today().isoformat())


def review(store: Store, topic: str, score: float, today: date | None = None) -> dict:
    """Reschedule `topic` after a quiz; creates the review if it is new."""
    today = today or date.today()
    row = find_review(store, topic) or store.insert("reviews", topic=topic.strip(), due=today.isoformat())
    ease, interval, reps = next_schedule(row["ease"], row["interval_days"], row["reps"], score)
    lapses = row["lapses"] + (score < PASS)
    return store.update("reviews", row["id"], ease=round(ease, 2), interval_days=interval, reps=reps,
                        lapses=lapses, last_score=float(score), due=(today + timedelta(days=interval)).isoformat())


def due(store: Store, today: date | None = None, limit: int = 10) -> dict:
    today = (today or date.today()).isoformat()
    rows = store.db.execute("SELECT * FROM reviews WHERE due <= ? ORDER BY due, lapses DESC LIMIT ?",
                            [today, limit]).fetchall()
    upcoming = store.db.execute("SELECT topic, due FROM reviews WHERE due > ? ORDER BY due LIMIT 5", [today]).fetchall()
    return {"today": today, "due": [dict(r) for r in rows], "upcoming": [dict(r) for r in upcoming]}


# ---- curriculum -------------------------------------------------------------------------


def load_curriculum(path: Path = CURRICULUM) -> list[dict]:
    return yaml.safe_load(path.read_text())["modules"]


def init_curriculum(store: Store, path: Path = CURRICULUM) -> dict:
    """Add every module to the reviews and every reading to the reading list. Safe to re-run."""
    added_topics, added_papers = 0, 0
    for i, m in enumerate(load_curriculum(path)):
        notes = "Objectives: " + " | ".join(m.get("objectives", [])) + f"\nExercise: {m.get('exercise', '')}"
        # Stagger the first quizzes: one new module per day instead of nine on day one.
        first = (date.today() + timedelta(days=i)).isoformat()
        added_topics += "already_scheduled" not in add_review(store, m["topic"], notes, due=first)
        for r in m.get("readings", []):
            if r.get("url") and store.list("papers", {"url": r["url"]}, limit=1):
                continue
            store.insert("papers", title=r["title"], authors=r.get("authors"), year=r.get("year"), url=r.get("url"),
                         tags=f"mas-core, {m['topic']}", status="to_read")
            added_papers += 1
    return {"topics_added": added_topics, "papers_added": added_papers}


# ---- scheduled jobs ---------------------------------------------------------------------


def jobs(config: dict) -> list[dict]:
    focus = (config.get("study") or {}).get("focus", "")
    owner = config.get("owner") or "the user"
    out = []
    for j in (config.get("study") or {}).get("schedule") or []:
        prompt = j["prompt"].replace("{focus}", focus).replace("{owner}", owner).strip()
        out.append({**j, "prompt": prompt, "sticky": bool(j.get("sticky"))})
    return out


def is_due(job: dict, now: datetime, last_run_day: str | None) -> bool:
    """Due once per day, at or after its time, on its days (default every day)."""
    days = [d.lower()[:3] for d in job.get("days") or DAYS]
    hour, _, minute = str(job["at"]).partition(":")
    at = now.replace(hour=int(hour), minute=int(minute or 0), second=0, microsecond=0)
    return DAYS[now.weekday()] in days and now >= at and last_run_day != now.date().isoformat()
