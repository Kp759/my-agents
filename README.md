# Dr.K's Agents

An open-source, **transparent** multi-agent system for PhD study, built with LangGraph. It runs on the AI subscriptions you already have (Claude, ChatGPT) and on free local models, with no API keys needed.

The shared goal of every agent: help you become a strong candidate for a research role at Anthropic, by **teaching you how things work in depth** and **building your research depth**. It is also meant for learning how multi-agent systems work: every routing decision, prompt, tool call, delegation, cost and memory is visible.

## The team

| Agent | What it does | Runs on |
|---|---|---|
| **@orchestrator** | The main agent: plans multi-step requests and delegates to the others | Claude Sonnet |
| **@explainer** | Explains concepts, papers and code in depth | Claude Sonnet |
| **@examiner** | Quizzes you, grades your answers and tracks weak topics | Codex (ChatGPT) |
| **@task_runner** | Manages your to-do list and drafts the deliverables | Claude Sonnet |
| **@experiment_tracker** | Lab notebook: logs runs, compares them, suggests the next experiment | Claude Haiku |
| **@phd_tracker** | Milestones, deadlines, advisor meetings, progress reviews | Claude Haiku |
| **@lit_surveyor** | Searches arXiv/OpenAlex, summarises papers, keeps your reading list | Claude Sonnet |
| **@trend_scout** | Finds rising research directions that fit you | Claude Sonnet |
| **@career_coach** | Reads real Anthropic job posts and maps your skill gaps | Claude Sonnet |
| **@coder** | Writes and edits code in your repos (with your approval) | Claude Sonnet |
| **@reviewer** | Independent critic of code, experiments and drafts | Codex (ChatGPT) |
| **@writer** | Paper sections, statements, emails; saves drafts | Claude Sonnet |
| **@friday** | General assistant for everyday questions: your calendar, free time, quick facts, news, reminders | Claude Haiku |

The Coder and the Reviewer run on **different model families** on purpose, so the review catches different mistakes.

## How a request flows

```
you ─► router (local Qwen3 8B, ~0.4 s, free) ─► agent ⇄ tools
                                                  │ delegate ▲ result
                                                  ▼          │
                                               other agent ⇄ tools
```

1. **Route:** a small local model picks one agent. Multi-step requests go to the orchestrator. `@agent` skips routing.
2. **Act:** the agent calls tools through a text protocol that works on every backend (a ReAct loop).
3. **Delegate:** an agent can hand a self-contained sub-task to another agent and get the answer back, like a function call.
4. **Ask:** risky actions pause the graph until you approve (LangGraph `interrupt`).
5. **Remember:** each agent keeps long-term memory, and a shared profile carries over between sessions.
6. **Trace:** every step's exact prompt, reply, tokens, latency and cost is recorded.

The dashboard's **How it works** tab explains each concept and links it to the code.

## Quickstart

Needs Python 3.11+, and any of: [Claude Code](https://claude.com/claude-code) (signed in), the [Codex CLI](https://github.com/openai/codex) (signed in with ChatGPT), or [Ollama](https://ollama.com) with `ollama pull qwen3:8b`.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/my-agents doctor              # which backends are reachable
.venv/bin/my-agents chat                # talk to the team; @writer, @coder… to pick an agent
.venv/bin/my-agents ui                  # dashboard + chat at http://127.0.0.1:8765
```

| Command | What it shows |
|---|---|
| `my-agents ask "…"` | One request, with every step live and its cost |
| `my-agents chat [--thread NAME]` | Multi-turn chat; saved, resume with the same `--thread` |
| `my-agents trace [ID]` | Replay a past request step by step |
| `my-agents board -f` | Live feed of what every agent is working on |
| `my-agents eod` | End-of-day report: what each agent did, how, and the cost |
| `my-agents ui` | **Chat with the agents in the browser** (approve actions there too), plus a live board, step-by-step timelines with exact prompts, team topology, memory and EOD |
| `my-agents telegram` | The Telegram bot (run `telegram setup` once first) |
| `my-agents calendar [show\|list\|add NAME\|remove NAME]` | Connect calendars for @friday (read-only iCal feeds) and see your agenda |
| `my-agents study [init\|due\|jobs\|run JOB]` | Load the MAS curriculum, see topics due for review, run a study job now |
| `my-agents bench run SUITE` | Benchmark team configurations on a task suite (`mas_core`, `injection`) |
| `my-agents bench list` / `show ID` | Past benchmark runs and their reports |
| `my-agents agents` / `tools` | Agents, their tools and delegates; tools and their safety flags |

## Telegram

Chat with your agents from your phone, approve actions with buttons, and get the end-of-day report every evening.

```bash
.venv/bin/my-agents telegram setup   # one time: create a bot with @BotFather, paste its token, press Start
.venv/bin/my-agents telegram         # keep this running (it only makes outbound connections)
```

In Telegram, just write; `@writer …` picks an agent. Commands: `/new`, `/status`, `/trace`, `/agents`, `/eod`, `/jobs`, `/run JOB`, `/due`, `/done`, `/help`. The bot answers only your Telegram account. The token is stored in `~/.config/my-agents/telegram.env` (owner-only, outside the repo). Telegram bot chats are not end-to-end encrypted.

## Study loop

A daily routine that runs itself through Telegram (`config.yaml → study.schedule`, edit freely):

| When | Agent | What you get |
|---|---|---|
| 08:00 daily | Trend Scout | The 3 most important new multi-agent papers or posts of the week, with why each matters in an interview |
| 20:30 daily | Examiner | A 3-question quiz on the topic most due for review; just reply (your answers go to the Examiner until `/done`) |
| Sun 18:00 | Career Coach | Current Anthropic roles, your top 3 gaps with this week's actions, and one mock interview question |

Quiz scores drive **spaced repetition**: a good answer pushes a topic further out (1 day, 3 days, then longer), a weak one brings it back tomorrow. `my-agents study init` loads a curriculum for LLM multi-agent systems: 10 modules from agent foundations to MAS safety plus one for your own research project, 23 readings with checked links (Wooldridge, ReAct, AutoGen, CAMEL, Generative Agents, debate, Anthropic's agent posts, MAS failure modes, prompt infection, multi-agent risks), and a hands-on exercise in this repository for each. See the dashboard's **Study** tab.

## Research: the benchmark harness

The system doubles as a testbed. `my-agents bench run SUITE` runs a suite of tasks with checkable answers under several configurations of the team and reports pass rate (with 95% Wilson intervals), cost, time, model and tool calls, delegations and tokens. Each trial gets a fresh in-memory store; network, GitHub, file and code tools are replaced by canned fixtures, so trials are repeatable and can't touch anything real.

- **`mas_core`**: when does a team beat a single agent, and at what cost? Configs: `router`, `orchestrator`, `orchestrator_shared_context` (delegates see the conversation), `solo_sonnet`/`solo_haiku`/`solo_qwen` (one agent with every tool), `team_qwen`.
- **`injection`**: how does prompt injection spread between agents, and what does each defence buy? Poisoned papers, pages, job posts, files, a fake user turn inside data and a forged orchestrator message, with clean controls. It measures whether an attack was attempted or executed, whether it hijacked the answer, and whether it propagated to an agent that never read the poison. Defences are switched off (labels, taint gate, egress gate) only inside the benchmark sandbox.

Results land in `data/bench/<run>/` with a `report.md` ready for a write-up. In the dashboard's **Bench** tab you can open every step of every trial. Suites are YAML in `bench/suites/`; add tasks and repeats to turn a pilot into a result.

## Research: the stop-signal lab

The current research question (see [docs/research/literature-scaling.md](docs/research/literature-scaling.md)):
*can adaptive multi-agent compute controllers trust agreement?* Controllers like HCP-MAD stop when agents
agree and add agents when they disagree, but agreement comes from the agents themselves. `my-agents lab run`
tests that on synthetic **hidden-profile** tasks (shared facts favour a trap; the right answer needs the private
facts of at least three members) in a 2×2 design (full or split information × honest team or one adversary).
Adversaries can fake agreement (`premature_stop`), never agree (`budget_exhaustion`), hide their decisive facts
(`withholding`), or be honest agents compromised by injected text (`injected`).

```bash
my-agents lab run --tasks 4                                  # pilot on local Qwen (free)
my-agents lab run --model claude:haiku --adversary budget_exhaustion --workers 4
my-agents lab list | show RUN
```

Besides accuracy it reports unsafe stops (wrong when the controller chose to stop), premature stops (stopped
before the surfaced evidence determined the answer), coverage of private facts at stop, compute amplification
caused by the adversary, and the information × adversary interaction. The dashboard's **Lab** tab shows every
trial's discussion and every exact prompt.

## Friday and your calendar

**@friday** is the everyday assistant: "what's on tomorrow?", "when am I free on Thursday afternoon?", "when is my next advisor meeting?", "remind me to submit the form on Friday", plus quick facts, news and general questions that no specialist covers. It runs on Claude Haiku (fast and cheap), and the router sends it everyday questions automatically.

Connect calendars read-only through their private **iCal address** (no API keys, no OAuth):

```bash
.venv/bin/my-agents calendar add personal   # paste the address into a hidden prompt
.venv/bin/my-agents calendar add njit       # add as many as you like
.venv/bin/my-agents calendar                # this week's agenda; `calendar list`, `calendar remove NAME`
```

Where to find the address: **Google Calendar** → Settings → your calendar → *Integrate calendar* → *Secret address in iCal format*; **iCloud** → share the calendar → *Public Calendar*; **Outlook** → Settings → *Shared calendars* → *Publish*. A local `.ics` file works too. The addresses are secrets, so they are stored owner-only in `~/.config/my-agents/calendars.txt` and never shown to a model, a trace or an error.

Friday's calendar tools: `calendar_events` (a date range, recurring events expanded, optional text filter) and `free_slots` (free gaps in a day's working hours). It also sees your tasks and milestones, adds reminders as tasks, and can't create or change calendar events (it gives you the details to paste instead). The orchestrator has the same calendar tools for planning your week. Event details are sent to the model that runs Friday; set `backend: ollama` in `agents/friday.yaml` to keep them on your machine.

## Up-to-date information

Every model stops learning at its training cutoff, so agents never rely on memory for anything that changes. Each prompt starts with **today's date** and rules to look things up and cite sources with dates (`config.yaml → freshness`). The lookups are our own tools, so they work the same on Claude, Codex and the local Qwen:

| Tool | Source |
|---|---|
| `web_search` | DuckDuckGo, with Bing as the fallback (no keys); `recency: day/week/month/year`. By default it also reads the top 3 pages and returns the passages that match the query (`deep: false` for snippets only), so even a small local model answers from real pages |
| `web_fetch` | Any public web page as text, with its publication date |
| `tech_news` | Hacker News stories with dates and points |
| `package_info` | Latest version and release dates from PyPI or npm |
| `search_papers`, `trending_papers` | arXiv / OpenAlex, Hugging Face daily papers |
| `github_*` | Repos, releases, issues, code |

**Search first, then verify** (`config.yaml → grounding: auto`). Agents don't have to remember to search: when an agent that has `web_search` starts on a question, the local router model decides (free, under a second) whether it needs facts about the world and writes a query. The graph runs the search and gives the results to the agent, which answers in three steps: from the sources, then checked against its own knowledge ("confirmed", or what disagrees and which it trusts), then the answer with dated citations, marking anything unverified. Personal questions (calendar, tasks), actions, quizzes and small talk skip the search. `grounding: off` leaves it to each agent.

Each lookup shows in the timeline, so you can see where an answer came from.

## GitHub

Agents use the `gh` CLI (`gh auth login`). They can search repos (for example, official code for a paper), read READMEs, files, issues, PRs and diffs, and clone repos into `~/Downloads/agent-repos`. The Coder can push a feature branch and open a **draft** PR, and the Coder and Reviewer can comment. Every write asks you first and only works on repos in `github.write_repos` (default: your own).

## Configure

- **Agents:** one YAML file each in `agents/` (backend, model, role prompt, tools, delegates). Copy `agents/_TEMPLATE.yaml` to add one.
- **`config.yaml`:**
  - `accounts`: which Claude and Codex accounts the agents use (`CLAUDE_CONFIG_DIR` / `CODEX_HOME` folders). Pinned so the agents use the same accounts whichever terminal starts them; `my-agents doctor` shows them.
  - `owner`: whose team this is (`Dr.K`). Agents address the owner by this name, and the dashboard and reports use it.
  - `router.entry`: `router` (a cheap local router) or `orchestrator` (every message goes through the main agent).
  - `mission`: the goal prepended to every agent's prompt.
  - `freshness`: today's date (`{today}`) and the "check, don't remember" rules, added after the mission.
  - `eod`: which model writes the end-of-day prose.
  - `pricing`: per-token prices, used to estimate cost for backends that don't report it.
  - `security.allowed_roots`: the folders file and code tools may touch.
  - `github.write_repos`: the repos agents may open issues, comments and PRs on (always with your approval).

**Cost:** Claude's CLI reports the API-equivalent price of each call. On a subscription you aren't billed this, but it's the right number for comparing designs. Codex reports tokens only.

## Security

Agents read untrusted text (papers, job posts, files), so the system defends against prompt injection with:
- labelled untrusted content;
- a taint gate: after reading outside content, every write needs your approval;
- an egress allowlist (`web_fetch` may read any public page, but after outside content a URL the agent made up needs your approval);
- a path policy that never exposes secrets;
- secret redaction;
- locked-down CLI backends;
- a hardened local dashboard.

See [docs/SECURITY.md](docs/SECURITY.md).

## Develop

```bash
.venv/bin/pytest            # offline: graph, delegation, approvals, memory, security attacks
.venv/bin/pytest -m live    # calls the real backends (uses a little quota)
langgraph dev               # LangGraph Studio (pip install "langgraph-cli[inmem]")
```

## License

MIT
