# CLAUDE.md: My Agents

A personal, transparent multi-agent system for PhD study, on LangGraph. The owner's goals: learn how MAS work (so nothing may become a black box), get a research job at Anthropic, and use this as a testbed for their MAS research (`Kp759/multi-agent-system`).

Remote: `github.com/Kp759/my-agents`. After every commit, `git push`.

## Rules
- **No API keys.** Backends are the `claude` and `codex` CLIs (subscriptions) and Ollama (local). Don't add paid-API backends unless asked.
- **Transparency first.** Every new step type must be traced (`tracing.py`), shown in `report.format_row` and in the dashboard timeline (`ui/index.html → stepCard`), and explained in the dashboard's "How it works" tab when it adds a concept.
- **Security.** New tools must set `writes`, `untrusted` and `approval` correctly. Network access goes through `tools/http.py` (allowlist) and file access through `guardrails.check_path`. Add an attack test to `tests/test_security.py` for any new capability. See `docs/SECURITY.md`.
- **Agents.** One YAML file per agent in `agents/`. Adding one also means updating the README table and the orchestrator's `delegates` list.
- **Benchmarks.** Suites live in `bench/suites/*.yaml`. New tools must work inside `bench.bench_toolbox`: store-backed tools run for real on the trial's in-memory store, and anything with outside effects must be stubbed there.
- **Tests.** `pytest` (offline, fake backends) must pass. Use `pytest -m live` for real backends; use a throwaway `MY_AGENTS_DATA` for manual live runs so the user's real memory isn't polluted.
- **LangGraph gotcha.** Never put `run_id` in `configurable`: LangGraph reserves it. We use `request_id`.
