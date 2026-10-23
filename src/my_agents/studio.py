"""Entry point for LangGraph Studio (`langgraph dev`), which draws the graph and lets
you step through runs. Studio supplies its own checkpointer, so none is set here.

    pip install "langgraph-cli[inmem]" && langgraph dev
"""

from .agents import load_agents, load_router_config
from .graph import build_graph
from .tracing import Tracer

graph = build_graph(load_agents(), load_router_config(), Tracer(), checkpointer=None)
