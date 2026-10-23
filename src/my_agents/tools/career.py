"""Job postings from public Greenhouse job boards (Anthropic uses Greenhouse)."""

from __future__ import annotations

import html
import re

from . import Tool
from .http import get_json

BOARD = "https://boards-api.greenhouse.io/v1/boards/{company}/jobs"


def job_postings(company: str = "anthropic", keyword: str | None = None, limit: int = 20) -> dict:
    jobs = get_json(BOARD.format(company=company.lower()))["jobs"]
    words = (keyword or "").lower().split()
    hits = [j for j in jobs if all(w in j["title"].lower() for w in words)]
    return {
        "company": company,
        "total_open": len(jobs),
        "matching": len(hits),
        "jobs": [
            {"id": j["id"], "title": j["title"], "location": j["location"]["name"], "url": j["absolute_url"]}
            for j in hits[: max(1, min(int(limit), 50))]
        ],
    }


def job_description(job_id: int, company: str = "anthropic") -> dict:
    j = get_json(f"{BOARD.format(company=company.lower())}/{int(job_id)}", {"content": "true"})
    text = re.sub(r"<[^>]+>", " ", html.unescape(j.get("content", "")))
    text = re.sub(r"\s+", " ", text).strip()
    return {"title": j["title"], "url": j["absolute_url"], "description": text[:8000]}


def tools() -> list[Tool]:
    return [
        Tool("job_postings", "List open roles on a company's Greenhouse job board, filtered by words in the title "
             "(e.g. keyword='research engineer').",
             {"company": "str, Greenhouse board name, default anthropic, optional",
              "keyword": "str, optional", "limit": "int, optional"}, job_postings, untrusted=True),
        Tool("job_description", "Full text of one job posting (requirements, responsibilities).",
             {"job_id": "int", "company": "str, optional"}, job_description, untrusted=True),
    ]
