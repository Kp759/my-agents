"""Which Claude and Codex accounts the agents use.

The agents run the `claude` and `codex` CLIs as child processes. Each CLI picks
its account from a config folder: CLAUDE_CONFIG_DIR (default ~/.claude) and
CODEX_HOME (default ~/.codex). Without pinning, the agents would use whatever the
launching terminal has set. `accounts:` in config.yaml pins them, so every agent
call goes to the same accounts regardless of where `my-agents` is started.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1].parent / "config.yaml"


def _accounts() -> dict:
    try:
        return (yaml.safe_load(CONFIG.read_text()) or {}).get("accounts") or {}
    except FileNotFoundError:
        return {}


def claude_env() -> dict:
    env = dict(os.environ)
    folder = _accounts().get("claude_config_dir")
    if folder:
        env["CLAUDE_CONFIG_DIR"] = str(Path(folder).expanduser())
    # Never let a stray API key or cloud setting silently switch billing away from the plan.
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"):
        env.pop(var, None)
    return env


def codex_env() -> dict:
    env = dict(os.environ)
    folder = _accounts().get("codex_home")
    if folder:
        env["CODEX_HOME"] = str(Path(folder).expanduser())
    env.pop("OPENAI_API_KEY", None)  # use the ChatGPT sign-in, not an API key
    return env


def describe() -> list[str]:
    """Human-readable account lines for `my-agents doctor`."""
    lines = []
    env = claude_env()
    try:
        out = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True, env=env, timeout=30)
        info = json.loads(out.stdout)
        lines.append(f"claude: {info.get('email', '?')} ({info.get('subscriptionType', '?')} plan, "
                     f"{info.get('authMethod', '?')}) · {env.get('CLAUDE_CONFIG_DIR', '~/.claude (default)')}")
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        lines.append(f"claude: could not read account ({e})")
    env = codex_env()
    try:
        out = subprocess.run(["codex", "login", "status"], capture_output=True, text=True, env=env, timeout=30)
        status = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else "?"
        lines.append(f"codex:  {status} · {env.get('CODEX_HOME', '~/.codex (default)')}")
    except (OSError, subprocess.TimeoutExpired) as e:
        lines.append(f"codex:  could not read account ({e})")
    return lines
