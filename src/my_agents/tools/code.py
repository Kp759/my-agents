"""Code tools for the Coder and Reviewer.

Reading is free. Changing code (run_coding_task) asks the user first, only works
inside a git repository (so every change shows up in `git diff` and can be
reverted), and hands the job to Claude Code or Codex running in that repository.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from ..guardrails import check_path
from . import Tool

MAX_CHARS = 20_000
# Codex features a code-editing job never needs; each is a way out of the sandbox.
CODEX_LOCKDOWN = ["--disable", "browser_use", "--disable", "browser_use_external",
                  "--disable", "computer_use", "--disable", "apps"]


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise ValueError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def _repo(path: str) -> Path:
    repo = check_path(path)
    if not repo.is_dir():
        raise ValueError(f"not a directory: {repo}")
    _git(repo, "rev-parse", "--is-inside-work-tree")
    return repo


def _cap(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f"\n… [truncated, {len(text)} chars total]"


def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
    p = check_path(path)
    if p.stat().st_size > 2_000_000:
        raise ValueError(f"{p} is larger than 2 MB")
    lines = p.read_text(errors="replace").splitlines()
    start = max(1, int(start_line))
    chunk = lines[start - 1 : start - 1 + min(int(max_lines), 1000)]
    body = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start))
    return _cap(f"{p} (lines {start}-{start + len(chunk) - 1} of {len(lines)})\n{body}")


def list_files(repo: str, pattern: str | None = None) -> str:
    files = _git(_repo(repo), "ls-files", *([pattern] if pattern else [])).splitlines()
    return _cap("\n".join(files[:500]) + (f"\n… {len(files) - 500} more" if len(files) > 500 else ""))


def git_diff(repo: str, staged: bool = False, base: str | None = None) -> str:
    r = _repo(repo)
    args = ["diff"] + (["--staged"] if staged else []) + ([base] if base else [])
    stat = _git(r, *args, "--stat")
    return _cap(stat + "\n" + _git(r, *args)) if stat.strip() else "no changes"


def run_coding_task(repo: str, instructions: str, engine: str = "claude") -> dict:
    r = _repo(repo)
    before = _git(r, "diff", "--stat")
    if engine == "claude":
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--permission-mode", "acceptEdits",
             "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep", "--no-session-persistence"],
            input=instructions, capture_output=True, text=True, cwd=r, timeout=1800,
        )
        try:
            summary = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            summary = proc.stdout[-2000:] or proc.stderr[-2000:]
    elif engine == "codex":
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "last.txt"
            proc = subprocess.run(
                ["codex", "exec", "--ephemeral", "--sandbox", "workspace-write", "--color", "never",
                 *CODEX_LOCKDOWN, "-C", str(r), "-o", str(out), "-"],
                input=instructions, capture_output=True, text=True, timeout=1800,
            )
            summary = out.read_text() if out.exists() else proc.stderr[-2000:]
    else:
        raise ValueError("engine must be 'claude' or 'codex'")
    return {
        "engine": engine,
        "exit_code": proc.returncode,
        "summary": summary[:4000],
        "diff_stat_before": before.strip() or "clean",
        "diff_stat_after": _git(r, "diff", "--stat").strip() or "no changes",
    }


def tools() -> list[Tool]:
    return [
        Tool("read_file", "Read a text file with line numbers.",
             {"path": "str", "start_line": "int, optional", "max_lines": "int, max 1000, optional"}, read_file,
             untrusted=True),
        Tool("list_files", "List files tracked in a git repository.",
             {"repo": "str, path", "pattern": "str, git pathspec like '*.py', optional"}, list_files, untrusted=True),
        Tool("git_diff", "Show uncommitted changes in a git repository (or against a base ref).",
             {"repo": "str, path", "staged": "bool, optional", "base": "str, e.g. main, optional"}, git_diff,
             untrusted=True),
        Tool("run_coding_task", "Have a coding agent edit files in a git repository: it reads the code, makes the "
             "change and reports back. Give precise instructions. Asks the user first. Shell commands are not "
             "allowed for the claude engine.",
             {"repo": "str, path to a git repository", "instructions": "str",
              "engine": "str, claude | codex, optional"}, run_coding_task, approval=True, writes=True),
    ]
