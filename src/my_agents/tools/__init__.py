"""Tools agents can call. Each agent lists the tools it may use in its YAML spec.

The CLI backends have no native function calling, so tools are called through a
small text protocol (see `my_agents.toolcall`). That works with every backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..store import Store


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # arg name -> "type, meaning" (append "optional" if not required)
    fn: Callable[..., object]
    approval: bool = False  # always pause the graph and ask the user before running
    writes: bool = False  # changes stored state; needs approval after untrusted content was read
    untrusted: bool = False  # returns outside content (web, files) that may contain prompt injection
    egress: bool = False  # sends its arguments to any host (a URL can carry data out); gated after untrusted content

    def required(self) -> set[str]:
        return {k for k, v in self.params.items() if "optional" not in v}

    def run(self, args: dict) -> object:
        unknown = set(args) - set(self.params)
        missing = self.required() - set(args)
        if unknown or missing:
            raise ValueError(
                f"bad arguments for {self.name}: unknown={sorted(unknown)} missing={sorted(missing)}; "
                f"expected {self.params}"
            )
        return self.fn(**args)


def make_toolbox(store: Store) -> dict[str, Tool]:
    from . import calendar, code, career, github, memory, research, web, writing

    tools = [*memory.tools(store), *research.tools(), *career.tools(), *code.tools(), *writing.tools(),
             *github.tools(), *web.tools(), *calendar.tools()]
    return {t.name: t for t in tools}
