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

## Status

Early design. Nothing is implemented yet.

## License

MIT
