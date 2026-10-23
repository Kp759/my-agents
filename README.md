# My Agents

An open-source, model-agnostic ecosystem of personal agents for PhD study.
You can plug in any LLM provider (Claude, ChatGPT, local models, and others) and run a team of specialised agents.

## Planned agents

| Agent | Role |
|---|---|
| **Explainer** | Explains papers, concepts and code at the depth you ask for |
| **Examiner** | Asks you questions to test your understanding (spaced repetition, viva prep) |
| **Task Runner** | Works through your pending tasks |
| **Experiment Tracker** | Logs experiments, configs, results and what to try next |
| **PhD Tracker** | Tracks milestones, deadlines, meetings and overall progress |
| **Literature Surveyor** | Searches, summarises and organises related work |
| **Trend Scout** | Finds emerging research topics worth working on |
| **Career Coach** | Finds the skill gaps between you and a target role (e.g. research at Anthropic) and builds a plan to close them |

## Design goals

- **Model-agnostic**: one provider interface; you choose the model per agent.
- **Composable**: every agent shares one interface, so new agents are easy to add.
- **Shared memory**: agents read and write a common store (tasks, experiments, papers, notes).
- **Local-first**: your data stays on your machine unless you choose otherwise.

## How it works

```
you ──► router (local model) ──► chosen agent ──► reply
               LangGraph graph; every call is traced to data/traces/*.jsonl
```

Agents run on **backends** that use tools you already have, so no API keys are needed:

| Backend | What it calls | Who pays |
|---|---|---|
| `claude-cli` | `claude -p` (Claude Code, headless) | your Claude subscription |
| `codex-cli` | `codex exec` (Codex CLI, signed in with ChatGPT) | your ChatGPT subscription |
| `ollama` | a local model through Ollama | free |

Each agent is a YAML file in `agents/` that sets its backend, model and system prompt.
To add an agent, copy `agents/_TEMPLATE.yaml`.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/my-agents doctor            # which backends are reachable
.venv/bin/my-agents agents            # list agents
.venv/bin/my-agents ask "explain KL divergence"
.venv/bin/my-agents ask -a examiner "quiz me on transformers"
.venv/bin/my-agents chat              # multi-turn; type @examiner or @explainer to switch
```

Tests: `pytest` (offline, uses a fake backend) and `pytest -m live` (calls the real backends).

## Status

| Agent | Status |
|---|---|
| Explainer | working (`claude-cli`, sonnet) |
| Examiner | working (`codex-cli`) |
| Others | planned |

## License

MIT
