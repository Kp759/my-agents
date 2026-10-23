"""A text protocol for tool calls, so tools work on backends without native function calling.

The model calls a tool by replying with only a fenced JSON block:

    ```json
    {"tool": "list_tasks", "args": {"status": "todo"}}
    ```

The result comes back as the next user message. Any reply without such a block
is the agent's final answer.
"""

from __future__ import annotations

import json
import re
from datetime import date

from .guardrails import redact, wrap_untrusted
from .tools import Tool

FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
MAX_RESULT_CHARS = 8_000

PROTOCOL = """
## Tools
{tools}

To call a tool, reply with ONLY a fenced JSON block, nothing before or after it:
```json
{{"tool": "<name>", "args": {{<arguments>}}}}
```
The result comes back in the next message. Call one tool at a time; use as few calls as you need.
Never invent tool results, IDs or citations. When you are done, reply to the user in plain
text with no JSON tool block. Today's date is {today}."""


def tools_prompt(tools: list[Tool]) -> str:
    if not tools:
        return ""
    lines = []
    for t in tools:
        args = ", ".join(f"{k} ({v})" for k, v in t.params.items()) or "no arguments"
        lines.append(f"- {t.name}: {t.description}\n  args: {args}")
    return PROTOCOL.format(tools="\n".join(lines), today=date.today().isoformat())


def parse_tool_call(text: str) -> dict | None:
    """Find a {"tool": ..., "args": ...} object in the reply.

    Fenced blocks are tried first; then any JSON object in the text, because models
    sometimes drop or mangle the fence (e.g. a closing ``` with no opening one).
    """
    candidates = FENCED.findall(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{" and '"tool"' in text[i:]:
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            candidates.append(obj)
    for raw in candidates:
        obj = raw
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            args = obj.get("args") or {}
            return {"tool": obj["tool"], "args": args if isinstance(args, dict) else {}}
    return None


def format_result(name: str, result: object, untrusted: bool = False) -> str:
    body = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False, indent=1)
    if len(body) > MAX_RESULT_CHARS:
        body = body[:MAX_RESULT_CHARS] + f"\n… [truncated, {len(body)} chars total]"
    body = redact(body)
    return f"[tool result: {name}]\n" + (wrap_untrusted(body) if untrusted else body)
