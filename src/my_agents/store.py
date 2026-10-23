"""Shared memory: one SQLite file that every agent reads and writes through tools.

Tables: tasks, experiments, milestones, notes, papers, quiz_results, skills, and
`board`, the shared feed where agents post what they are working on, and
`agent_memory`, each agent's long-term memory (agent='*' is the profile every agent shares).
Lives at `data/memory.sqlite` (override with MY_AGENTS_DATA); `data/` is gitignored.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .guardrails import secure_data_dir

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, details TEXT DEFAULT '',
    status TEXT DEFAULT 'todo', priority TEXT DEFAULT 'medium', due TEXT,
    result TEXT DEFAULT '', created TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, hypothesis TEXT DEFAULT '',
    config TEXT DEFAULT '{}', status TEXT DEFAULT 'planned', metrics TEXT DEFAULT '{}',
    notes TEXT DEFAULT '', next_steps TEXT DEFAULT '', created TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, due TEXT, status TEXT DEFAULT 'planned',
    notes TEXT DEFAULT '', created TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, kind TEXT DEFAULT 'note', content TEXT NOT NULL, created TEXT
);
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, authors TEXT DEFAULT '', year INTEGER,
    url TEXT UNIQUE, summary TEXT DEFAULT '', tags TEXT DEFAULT '', status TEXT DEFAULT 'to_read',
    created TEXT
);
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY, topic TEXT NOT NULL, score REAL, gaps TEXT DEFAULT '', created TEXT
);
CREATE TABLE IF NOT EXISTS board (
    id INTEGER PRIMARY KEY, agent TEXT NOT NULL, kind TEXT DEFAULT 'note', content TEXT NOT NULL,
    thread_id TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS agent_memory (
    id INTEGER PRIMARY KEY, agent TEXT NOT NULL, kind TEXT DEFAULT 'fact', content TEXT NOT NULL,
    created TEXT
);
CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY, title TEXT, created TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY, skill TEXT UNIQUE NOT NULL, level INTEGER, target INTEGER,
    evidence TEXT DEFAULT '', plan TEXT DEFAULT '', updated TEXT
);
"""

JSON_COLUMNS = {"config", "metrics"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str | None = None):
        if path is None:
            root = Path(os.environ.get("MY_AGENTS_DATA", DEFAULT_DATA_DIR))
            secure_data_dir(root)  # chmod 700: traces and memory are private
            path = root / "memory.sqlite"
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def insert(self, table: str, **fields) -> dict:
        fields = {k: _encode(k, v) for k, v in fields.items() if v is not None}
        cols = _columns(self.db, table)
        if "created" in cols:
            fields.setdefault("created", now())
        if "updated" in cols:
            fields.setdefault("updated", now())
        _check_columns(cols, fields, table)
        keys = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        cur = self.db.execute(f"INSERT INTO {table} ({keys}) VALUES ({marks})", list(fields.values()))
        self.db.commit()
        return self.get(table, cur.lastrowid)

    def update(self, table: str, id: int, **fields) -> dict:
        fields = {k: _encode(k, v) for k, v in fields.items() if v is not None}
        if not fields:
            raise ValueError("nothing to update")
        cols = _columns(self.db, table)
        if "updated" in cols:
            fields["updated"] = now()
        _check_columns(cols, fields, table)
        sets = ", ".join(f"{k} = ?" for k in fields)
        cur = self.db.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*fields.values(), id])
        self.db.commit()
        if cur.rowcount == 0:
            raise ValueError(f"no {table} row with id {id}")
        return self.get(table, id)

    def get(self, table: str, id: int) -> dict:
        row = self.db.execute(f"SELECT * FROM {table} WHERE id = ?", [id]).fetchone()
        if row is None:
            raise ValueError(f"no {table} row with id {id}")
        return _decode(row)

    def delete(self, table: str, id: int) -> None:
        cur = self.db.execute(f"DELETE FROM {table} WHERE id = ?", [id])
        self.db.commit()
        if cur.rowcount == 0:
            raise ValueError(f"no {table} row with id {id}")

    def list(self, table: str, where: dict | None = None, order: str = "id DESC", limit: int = 50) -> list[dict]:
        where = {k: v for k, v in (where or {}).items() if v is not None}
        _check_columns(_columns(self.db, table), where, table)
        clause = " AND ".join(f"{k} = ?" for k in where)
        sql = f"SELECT * FROM {table}" + (f" WHERE {clause}" if clause else "") + f" ORDER BY {order} LIMIT ?"
        return [_decode(r) for r in self.db.execute(sql, [*where.values(), limit])]

    def upsert_skill(self, skill: str, **fields) -> dict:
        row = self.db.execute("SELECT id FROM skills WHERE lower(skill) = lower(?)", [skill]).fetchone()
        if row:
            return self.update("skills", row["id"], **fields)
        return self.insert("skills", skill=skill, **fields)


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def _check_columns(cols: set[str], fields: dict, table: str) -> None:
    unknown = set(fields) - cols
    if unknown:
        raise ValueError(f"unknown {table} fields: {sorted(unknown)}")


def _encode(key: str, value):
    if key in JSON_COLUMNS and not isinstance(value, str):
        return json.dumps(value)
    return value


def _decode(row: sqlite3.Row) -> dict:
    out = dict(row)
    for k in JSON_COLUMNS & out.keys():
        try:
            out[k] = json.loads(out[k]) if out[k] else {}
        except json.JSONDecodeError:
            pass
    return out
