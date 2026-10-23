"""Agent specs: one YAML file per agent in `agents/`.

Adding an agent means adding a file; no code changes. Each spec names the backend
and model it runs on, so agents can mix Claude, ChatGPT and local models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .adapters import ChatAdapter, make_adapter

DEFAULT_AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class AgentSpec:
    name: str
    description: str  # what the router reads when choosing an agent
    system_prompt: str
    backend: str
    model: str | None = None
    options: dict = field(default_factory=dict)  # backend-specific, e.g. {think: false}
    tools: list[str] = field(default_factory=list)  # tool names from my_agents.tools
    delegates: list[str] = field(default_factory=list)  # agents this one may hand sub-tasks to
    board_context: bool = False  # show recent board posts in the system prompt

    def adapter(self) -> ChatAdapter:
        return make_adapter(self.backend, self.model, **self.options)


@dataclass
class RouterConfig:
    backend: str
    model: str | None
    default_agent: str
    options: dict = field(default_factory=dict)
    # "router": the local router picks an agent per message (cheap; multi-step requests go to the orchestrator).
    # "orchestrator": every message goes to the orchestrator, which delegates.
    entry: str = "router"
    orchestrator: str = "orchestrator"
    mission: str = ""  # prepended to every agent's system prompt
    owner: str = "the user"  # whose team this is; agents address them by this name


def load_agents(directory: Path = DEFAULT_AGENTS_DIR) -> dict[str, AgentSpec]:
    specs = {}
    for path in sorted(directory.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        data = yaml.safe_load(path.read_text())
        spec = AgentSpec(**data)
        if spec.name in specs:
            raise ValueError(f"duplicate agent name {spec.name!r} in {path}")
        specs[spec.name] = spec
    if not specs:
        raise ValueError(f"no agent specs found in {directory}")
    return specs


def load_router_config(path: Path = DEFAULT_CONFIG) -> RouterConfig:
    config = yaml.safe_load(path.read_text())
    data = config["router"]
    return RouterConfig(
        backend=data["backend"],
        model=data.get("model"),
        default_agent=data["default_agent"],
        options=data.get("options") or {},
        entry=data.get("entry", "router"),
        orchestrator=data.get("orchestrator", "orchestrator"),
        mission=(config.get("mission") or "").strip().replace("{owner}", config.get("owner") or "the user"),
        owner=config.get("owner") or "the user",
    )
