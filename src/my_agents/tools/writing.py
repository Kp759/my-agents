"""Drafts for the Writer: markdown files in `data/drafts/`. Names are sanitised, so a
draft can never be written outside that folder."""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import Tool

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _drafts_dir() -> Path:
    d = Path(os.environ.get("MY_AGENTS_DATA", DEFAULT_DATA_DIR)) / "drafts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(name: str) -> Path:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower().removesuffix(".md")).strip("-")[:80]
    if not slug:
        raise ValueError("draft name must contain letters or digits")
    return _drafts_dir() / f"{slug}.md"


def write_draft(name: str, content: str, append: bool = False) -> dict:
    p = _path(name)
    with open(p, "a" if append else "w") as f:
        f.write(content if not append else "\n\n" + content)
    return {"draft": p.stem, "path": str(p), "words": len(p.read_text().split())}


def read_draft(name: str) -> str:
    p = _path(name)
    if not p.exists():
        raise ValueError(f"no draft named {p.stem!r}; see list_drafts")
    return p.read_text()


def list_drafts() -> list[dict]:
    return [{"draft": p.stem, "words": len(p.read_text().split()), "modified": p.stat().st_mtime}
            for p in sorted(_drafts_dir().glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)]


def tools() -> list[Tool]:
    return [
        Tool("write_draft", "Save a draft (markdown) under a short name; overwrites unless append=true.",
             {"name": "str, e.g. related-work-v1", "content": "str", "append": "bool, optional"},
             write_draft, writes=True),
        Tool("read_draft", "Read a saved draft.", {"name": "str"}, read_draft, untrusted=True),
        Tool("list_drafts", "List saved drafts, newest first.", {}, list_drafts),
    ]
