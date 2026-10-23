"""Turn trace rows into views: a per-request timeline with cost, and the end-of-day report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

from .adapters import make_adapter
from .board import Board
from .store import Store
from .tracing import Tracer

NON_MODEL = ("tools", "delegate", "return", "approval")  # trace rows that are not model calls

# ---- cost ------------------------------------------------------------------------------


def row_cost(row: dict, pricing: dict | None = None) -> float | None:
    """Cost reported by the backend, else estimated from `pricing` in config.yaml, else None.

    pricing: {model or backend: {input: $/1M, cached_input: $/1M, output: $/1M}}
    """
    if row.get("cost_usd") is not None:
        return row["cost_usd"]
    if row.get("node") in NON_MODEL or row.get("input_tokens") is None:
        return None
    price = (pricing or {}).get(row.get("model") or "") or (pricing or {}).get(row.get("backend") or "")
    if not price:
        return None
    cached = row.get("cached_input_tokens") or 0
    fresh = (row.get("input_tokens") or 0) - cached
    return (fresh * price.get("input", 0) + cached * price.get("cached_input", price.get("input", 0))
            + (row.get("output_tokens") or 0) * price.get("output", 0)) / 1e6


def totals(rows: list[dict], pricing: dict | None = None) -> dict:
    model_rows = [r for r in rows if r.get("node") not in NON_MODEL and r.get("backend") != "none"]
    costs = [row_cost(r, pricing) for r in model_rows]
    unpriced = sorted({r["backend"] for r, c in zip(model_rows, costs) if c is None})
    ts = [datetime.fromisoformat(r["ts"]) for r in rows]
    first_start = ts[0] - timedelta(milliseconds=rows[0].get("latency_ms") or 0) if rows else None
    return {
        "model_calls": len(model_rows),
        "tool_calls": sum(r.get("node") == "tools" for r in rows),
        "delegations": sum(r.get("node") == "delegate" and r.get("ok") for r in rows),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in model_rows),
        "cached_input_tokens": sum(r.get("cached_input_tokens") or 0 for r in model_rows),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in model_rows),
        "cost_usd": sum(c for c in costs if c is not None),
        "unpriced_backends": unpriced,
        "wall_s": (ts[-1] - first_start).total_seconds() if rows else 0.0,
        "errors": sum(not r.get("ok", True) for r in rows),
    }


# ---- per-request timeline --------------------------------------------------------------


def _k(n) -> str:
    return "-" if n is None else (f"{n / 1000:.1f}k" if n >= 1000 else str(n))


def _secs(ms) -> str:
    return "" if ms is None else f"{ms / 1000:.1f}s"


def format_row(row: dict, pricing: dict | None = None) -> str:
    node = row.get("node")
    err = f"  ERROR: {row['error']}" if row.get("error") else ""
    if node == "route":
        model = row.get("model") or "-"
        return f"route        {model:<22} {_secs(row.get('latency_ms')):>6}  → @{row['agent']}  ({row.get('decision')}){err}"
    if node == "delegate":
        return f"  delegate   @{row.get('decision')}: {row.get('args')}{err}"
    if node == "return":
        return f"  return     @{row.get('decision')}: {row.get('output_preview')}"
    if node == "approval":
        return f"    approval {row.get('tool'):<22}  you said {row.get('decision')!r} ({row.get('reason')})"
    if node == "tools":
        out = row.get("output_preview") or ""
        return f"    tool     {row.get('tool'):<22} {_secs(row.get('latency_ms')):>6}  {row.get('args')}{err}" + (
            f"\n             ↳ {out[:160]}" if out else "")
    cost = row_cost(row, pricing)
    money = "" if cost is None else f"  ${cost:.4f}"
    indent = "  " * (row.get("depth") or 0)
    head = (f"{indent}@{row['agent']:<11} {(row.get('model') or row.get('backend') or '-'):<22} "
            f"{_secs(row.get('latency_ms')):>6}  {_k(row.get('input_tokens'))}→{_k(row.get('output_tokens'))} tok{money}")
    action = row.get("action") or ""
    body = f"\n{indent}             ↳ {(row.get('output_preview') or '')[:200]}" if action == "final answer" else f"  → {action}"
    return head + body + err


def format_totals(t: dict) -> str:
    money = f"${t['cost_usd']:.4f} API-equivalent"
    if t["unpriced_backends"]:
        money += f" (+ {', '.join(t['unpriced_backends'])}: tokens only, add prices in config.yaml)"
    return (f"total: {t['wall_s']:.1f}s · {t['model_calls']} model calls · {t['tool_calls']} tool calls · "
            f"{t['delegations']} delegations · {_k(t['input_tokens'])} in ({_k(t['cached_input_tokens'])} cached) / "
            f"{_k(t['output_tokens'])} out tokens · {money}" + (f" · {t['errors']} errors" if t["errors"] else ""))


# ---- end of day ------------------------------------------------------------------------


def _local_day(ts: str) -> date:
    return datetime.fromisoformat(ts).astimezone().date()


def day_rows(tracer: Tracer, day: date) -> list[dict]:
    rows = []
    for d in (day - timedelta(days=1), day, day + timedelta(days=1)):  # trace files are UTC days
        rows += tracer.load(d.isoformat())
    return [r for r in rows if _local_day(r["ts"]) == day]


def memory_changes(store: Store, day: date) -> dict:
    out = {}
    for table, cols in {"tasks": "id, title, status, result", "experiments": "id, name, status, metrics",
                        "milestones": "id, title, status, due", "papers": "id, title, status",
                        "quiz_results": "id, topic, score, gaps", "skills": "id, skill, level, target"}.items():
        stamp = "updated" if table in ("tasks", "experiments", "milestones", "skills") else "created"
        rows = store.db.execute(f"SELECT {cols}, {stamp} AS ts FROM {table}").fetchall()
        hits = [{k: r[k] for k in r.keys() if k != "ts"} for r in rows if r["ts"] and _local_day(r["ts"]) == day]
        if hits:
            out[table] = hits[:30]
    return out


def eod_facts(tracer: Tracer, store: Store, day: date, pricing: dict | None = None) -> dict:
    rows = day_rows(tracer, day)
    board_rows = [b for b in Board(store).recent(limit=200) if _local_day(b["created"]) == day]
    agents = defaultdict(lambda: {"requests": set(), "rows": [], "tools": Counter(), "delegated_to": Counter(),
                                  "delegated_from": Counter(), "models": Counter(), "worked_on": [], "results": []})
    for r in rows:
        a = r.get("agent")
        if not a or r.get("node") == "route":
            continue
        info = agents[a]
        if r["node"] == "tools":
            info["tools"][r.get("tool")] += 1
        elif r["node"] == "delegate" and r.get("ok"):
            target = (r.get("decision") or "→").split("→")[-1].strip()
            info["delegated_to"][target] += 1
            agents[target]["delegated_from"][a] += 1
        elif r["node"] not in ("return", "approval"):
            info["requests"].add(r["run_id"])
            info["rows"].append(r)
            info["models"][r.get("model") or r.get("backend")] += 1
    for b in board_rows:
        if b["kind"] == "start":
            agents[b["agent"]]["worked_on"].append(b["content"][:200])
        elif b["kind"] in ("done", "action"):
            agents[b["agent"]]["results"].append(f"{b['kind']}: {b['content'][:200]}")

    per_agent = {}
    for name, info in sorted(agents.items()):
        t = totals(info["rows"], pricing) if info["rows"] else totals([], pricing)
        per_agent[name] = {
            "requests": len(info["requests"]),
            "models": dict(info["models"]),
            "tools_used": dict(info["tools"]),
            "delegated_to": dict(info["delegated_to"]),
            "delegated_from": dict(info["delegated_from"]),
            "worked_on": info["worked_on"][:10],
            "results": info["results"][:10],
            "model_calls": t["model_calls"],
            "tokens_in": t["input_tokens"],
            "tokens_out": t["output_tokens"],
            "cost_usd": round(t["cost_usd"], 4),
            "model_time_s": round(sum((r.get("latency_ms") or 0) for r in info["rows"]) / 1000, 1),
            "errors": sum(not r.get("ok", True) for r in info["rows"]),
        }
    routes = Counter(r["agent"] for r in rows if r.get("node") == "route")
    return {
        "day": day.isoformat(),
        "user_requests": len({r["run_id"] for r in rows if r.get("node") == "route"}),
        "routed_to": dict(routes),
        "overall": totals(rows, pricing) if rows else totals([], pricing),
        "agents": per_agent,
        "memory_changes": memory_changes(store, day),
    }


EOD_PROMPT = """You write an end-of-day report for {owner}, a PhD student, about their team of AI agents.
You get exact facts as JSON. For each agent that did anything, write a "### @name" heading
followed by 2-4 sentences: what it worked on, how (tools, delegations, models), and what it
produced. Then a "### Overall" section: what got done, anything that failed, and one suggestion
for tomorrow. Use only the facts given; do not invent work. Plain markdown, concise."""


def eod_report(facts: dict, writer: dict | None, owner: str = "the user") -> str:
    o = facts["overall"]
    team = f"{owner}'s agents" if owner != "the user" else "agents"
    lines = [f"# End-of-day report: {team}, {facts['day']}", "",
             f"{facts['user_requests']} requests · {o['model_calls']} model calls · {o['tool_calls']} tool calls · "
             f"{o['delegations']} delegations · {_k(o['input_tokens'])} in / {_k(o['output_tokens'])} out tokens · "
             f"${o['cost_usd']:.4f} API-equivalent"
             + (f" (+ unpriced: {', '.join(o['unpriced_backends'])})" if o["unpriced_backends"] else ""), "",
             "| Agent | Requests | Model calls | Tools used | Delegated to | Tokens in/out | Cost | Model time |",
             "|---|---|---|---|---|---|---|---|"]
    for name, a in facts["agents"].items():
        tools = ", ".join(f"{k}×{v}" for k, v in a["tools_used"].items()) or "-"
        dele = ", ".join(f"@{k}×{v}" for k, v in a["delegated_to"].items()) or "-"
        lines.append(f"| @{name} | {a['requests']} | {a['model_calls']} | {tools} | {dele} | "
                     f"{_k(a['tokens_in'])}/{_k(a['tokens_out'])} | ${a['cost_usd']:.4f} | {a['model_time_s']}s |")
    if facts["memory_changes"]:
        lines += ["", "**Memory changes today:** " + ", ".join(
            f"{len(v)} {k}" for k, v in facts["memory_changes"].items())]

    narrative = None
    if writer and facts["agents"]:
        try:
            adapter = make_adapter(writer["backend"], writer.get("model"), **(writer.get("options") or {}))
            narrative = adapter.complete(EOD_PROMPT.format(owner=owner),
                                         [{"role": "user", "content": json.dumps(facts, default=str)}]).text
        except Exception as e:  # the table above is still useful without the prose
            narrative = f"_(narrative unavailable: {e})_"
    if narrative:
        lines += ["", "## What each agent did", "", narrative.strip()]
    elif not facts["agents"]:
        lines += ["", "No agent activity today."]
    return "\n".join(lines) + "\n"
