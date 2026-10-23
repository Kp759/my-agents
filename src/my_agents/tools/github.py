"""GitHub through the `gh` CLI (uses your `gh auth login`; no token handled here).

Reading: any repo you can see. Everything read from GitHub is untrusted: anyone
can write an issue, README or PR description, so it may contain prompt injection.

Writing (issues, comments, pull requests) is outward-facing and public, so:
- it always asks you first (approval);
- it only targets repos matching `github.write_repos` in config.yaml (default your own);
- pull requests are opened as drafts, never from the default branch;
- posts are signed so readers know an agent drafted them.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import yaml

from ..guardrails import GuardrailError, check_path
from . import Tool

CONFIG = Path(__file__).resolve().parents[3] / "config.yaml"
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}$")
REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,200}$")
MAX_CHARS = 20_000


def _config() -> dict:
    try:
        data = yaml.safe_load(CONFIG.read_text()) or {}
    except FileNotFoundError:
        data = {}
    return {"github": data.get("github") or {}, "owner": data.get("owner") or ""}


def _repo(repo: str) -> str:
    repo = repo.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    if not REPO_RE.match(repo):
        raise ValueError(f"not a valid owner/name repository: {repo!r}")
    return repo


def _ref(ref: str | None) -> str | None:
    if ref is not None and not REF_RE.match(ref):
        raise ValueError(f"not a valid branch, tag or commit: {ref!r}")
    return ref


def _path(path: str) -> str:
    path = path.strip().lstrip("/")
    if ".." in path.split("/") or path.startswith("-"):
        raise ValueError(f"invalid path: {path!r}")
    return path


def _number(n) -> int:
    n = int(n)
    if n <= 0:
        raise ValueError("issue/PR number must be positive")
    return n


def _check_writable(repo: str) -> None:
    allowed = _config()["github"].get("write_repos") or []
    if not any(fnmatch.fnmatch(repo.lower(), pat.lower()) for pat in allowed):
        raise GuardrailError(f"blocked: agents may only write to {allowed} (github.write_repos in config.yaml), "
                             f"not {repo}")


def _signed(body: str) -> str:
    owner = _config()["owner"]
    team = f"{owner}'s agents" if owner else "an AI agent"
    return f"{body.rstrip()}\n\n---\n_Drafted by {team} and approved by a human before posting._"


def _gh(*args: str, cwd: str | None = None, timeout: int = 60) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if proc.returncode != 0:
        raise ValueError(f"gh {args[0]} failed: {(proc.stderr or proc.stdout).strip()[:500]}")
    out = proc.stdout
    return out if len(out) <= MAX_CHARS else out[:MAX_CHARS] + f"\n… [truncated, {len(out)} chars total]"


def _gh_json(*args: str):
    return json.loads(_gh(*args) or "null")


# ---- reading --------------------------------------------------------------------------


def github_my_repos(limit: int = 30) -> list[dict]:
    return _gh_json("repo", "list", "--limit", str(min(int(limit), 100)), "--json",
                    "nameWithOwner,description,isPrivate,primaryLanguage,stargazerCount,updatedAt,url")


def github_search_repos(query: str, limit: int = 10, sort: str = "stars") -> list[dict]:
    if sort not in ("stars", "updated", "best-match"):
        raise ValueError("sort must be stars, updated or best-match")
    args = ["search", "repos", "--limit", str(min(int(limit), 30)),
            "--json", "fullName,description,stargazersCount,updatedAt,url,language"]
    if sort != "best-match":
        args += ["--sort", sort]
    # One argument per word: gh treats a single argument with spaces as an exact phrase.
    return _gh_json(*args, "--", *query.split())


def github_repo(repo: str) -> dict:
    repo = _repo(repo)
    info = _gh_json("repo", "view", repo, "--json",
                    "nameWithOwner,description,stargazerCount,forkCount,primaryLanguage,repositoryTopics,"
                    "licenseInfo,defaultBranchRef,updatedAt,url")
    try:
        info["readme"] = _gh("api", f"repos/{repo}/readme", "-H", "Accept: application/vnd.github.raw")[:6000]
    except ValueError:
        info["readme"] = None
    return info


def github_list_files(repo: str, path: str = "", ref: str | None = None) -> list[dict]:
    repo, path, ref = _repo(repo), _path(path), _ref(ref)
    endpoint = f"repos/{repo}/contents/{quote(path)}" + (f"?ref={quote(ref)}" if ref else "")
    items = _gh_json("api", endpoint)
    if isinstance(items, dict):
        items = [items]
    return [{"path": i["path"], "type": i["type"], "size": i.get("size")} for i in items]


def github_read_file(repo: str, path: str, ref: str | None = None) -> str:
    repo, path, ref = _repo(repo), _path(path), _ref(ref)
    endpoint = f"repos/{repo}/contents/{quote(path)}" + (f"?ref={quote(ref)}" if ref else "")
    return _gh("api", endpoint, "-H", "Accept: application/vnd.github.raw")


def github_issues(repo: str, state: str = "open", limit: int = 20) -> list[dict]:
    if state not in ("open", "closed", "all"):
        raise ValueError("state must be open, closed or all")
    return _gh_json("issue", "list", "-R", _repo(repo), "--state", state, "--limit", str(min(int(limit), 100)),
                    "--json", "number,title,state,author,labels,updatedAt,url")


def github_issue(repo: str, number: int) -> dict:
    data = _gh_json("issue", "view", str(_number(number)), "-R", _repo(repo), "--json",
                    "number,title,body,state,author,labels,comments,url")
    data["comments"] = [{"author": c["author"]["login"], "body": c["body"][:2000]} for c in data["comments"][-15:]]
    return data


def github_prs(repo: str, state: str = "open", limit: int = 20) -> list[dict]:
    if state not in ("open", "closed", "merged", "all"):
        raise ValueError("state must be open, closed, merged or all")
    return _gh_json("pr", "list", "-R", _repo(repo), "--state", state, "--limit", str(min(int(limit), 100)),
                    "--json", "number,title,state,author,headRefName,baseRefName,isDraft,updatedAt,url")


def github_pr(repo: str, number: int, diff: bool = False) -> dict:
    repo, number = _repo(repo), _number(number)
    data = _gh_json("pr", "view", str(number), "-R", repo, "--json",
                    "number,title,body,state,isDraft,author,headRefName,baseRefName,additions,deletions,"
                    "files,reviewDecision,url")
    if diff:
        data["diff"] = _gh("pr", "diff", str(number), "-R", repo)
    return data


# ---- writing (always approved by the user) -------------------------------------------


def github_create_issue(repo: str, title: str, body: str) -> dict:
    repo = _repo(repo)
    _check_writable(repo)
    url = _gh("issue", "create", "-R", repo, "--title", title, "--body", _signed(body)).strip()
    return {"created": url}


def github_comment(repo: str, number: int, body: str) -> dict:
    repo, number = _repo(repo), _number(number)
    _check_writable(repo)
    out = _gh_json("api", f"repos/{repo}/issues/{number}/comments", "-f", f"body={_signed(body)}")
    return {"commented": out.get("html_url")}


def github_clone(repo: str) -> dict:
    repo = _repo(repo)
    base = check_path(Path(_config()["github"].get("clone_dir") or "~/Downloads/agent-repos").expanduser())
    target = base / repo.replace("/", "__")
    if target.exists():
        return {"already_cloned": str(target)}
    base.mkdir(parents=True, exist_ok=True)
    _gh("repo", "clone", repo, str(target), "--", "--depth", "1", timeout=600)
    return {"cloned": str(target), "note": "shallow clone (latest commit only); code was not run"}


def github_create_pr(repo_path: str, title: str, body: str, base: str | None = None) -> dict:
    local = check_path(repo_path)

    def git(*args):
        proc = subprocess.run(["git", "-C", str(local), *args], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise ValueError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
        return proc.stdout.strip()

    remote = git("remote", "get-url", "origin")
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", remote)
    if not m:
        raise ValueError(f"origin is not a GitHub repository: {remote}")
    repo = _repo(m.group(1))
    _check_writable(repo)
    branch = git("branch", "--show-current")  # empty when HEAD is detached
    default = _gh_json("repo", "view", repo, "--json", "defaultBranchRef")["defaultBranchRef"]["name"]
    if branch in ("", default):
        raise GuardrailError(f"blocked: create a feature branch first; will not open a PR from {branch!r}")
    if git("status", "--porcelain"):
        raise ValueError("uncommitted changes: commit them first so the PR contains exactly what was reviewed")
    git("push", "-u", "origin", branch)
    args = ["pr", "create", "--draft", "--title", title, "--body", _signed(body), "--head", branch]
    if base:
        args += ["--base", _ref(base)]
    url = _gh(*args, cwd=str(local)).strip().splitlines()[-1]
    return {"draft_pr": url, "branch": branch, "repo": repo}


def tools() -> list[Tool]:
    repo = "str, owner/name"
    return [
        Tool("github_my_repos", "List the owner's own GitHub repositories.", {"limit": "int, optional"},
             github_my_repos, untrusted=True),
        Tool("github_search_repos", "Search GitHub repositories (e.g. code released with a paper).",
             {"query": "str", "limit": "int, max 30, optional", "sort": "str, stars | updated | best-match, optional"},
             github_search_repos, untrusted=True),
        Tool("github_repo", "Details and README of a repository.", {"repo": repo}, github_repo, untrusted=True),
        Tool("github_list_files", "List files in a repository folder.",
             {"repo": repo, "path": "str, folder, optional", "ref": "str, branch/tag/commit, optional"},
             github_list_files, untrusted=True),
        Tool("github_read_file", "Read a file from a repository on GitHub.",
             {"repo": repo, "path": "str", "ref": "str, optional"}, github_read_file, untrusted=True),
        Tool("github_issues", "List issues of a repository.",
             {"repo": repo, "state": "str, open | closed | all, optional", "limit": "int, optional"},
             github_issues, untrusted=True),
        Tool("github_issue", "Read one issue with its recent comments.", {"repo": repo, "number": "int"},
             github_issue, untrusted=True),
        Tool("github_prs", "List pull requests of a repository.",
             {"repo": repo, "state": "str, open | closed | merged | all, optional", "limit": "int, optional"},
             github_prs, untrusted=True),
        Tool("github_pr", "Read one pull request (files changed; set diff=true for the full diff).",
             {"repo": repo, "number": "int", "diff": "bool, optional"}, github_pr, untrusted=True),
        Tool("github_clone", "Shallow-clone a repository into the agents' clone folder so code tools can read it.",
             {"repo": repo}, github_clone, writes=True),
        Tool("github_create_issue", "Open an issue on one of the owner's repositories. Public. Asks the owner first.",
             {"repo": repo, "title": "str", "body": "str, markdown"}, github_create_issue, approval=True, writes=True),
        Tool("github_comment", "Comment on an issue or pull request in one of the owner's repositories. Public. "
             "Asks the owner first.", {"repo": repo, "number": "int", "body": "str, markdown"},
             github_comment, approval=True, writes=True),
        Tool("github_create_pr", "Push the current feature branch of a local repository and open a DRAFT pull "
             "request. Commit first; never from the default branch. Public. Asks the owner first.",
             {"repo_path": "str, local path", "title": "str", "body": "str, markdown",
              "base": "str, target branch, optional"}, github_create_pr, approval=True, writes=True),
    ]
