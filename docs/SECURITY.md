# Security

This system lets language models read the internet and your files and act on your data, so its main risks are specific to LLM agents. Below are the threats, the defence against each, and where each defence lives in the code. `tests/test_security.py` turns every row into an attack that must fail.

## Threat model

**What we protect:** your secrets (keys, SSH keys, credentials), your data (tasks, experiments, memory, drafts, code), and the integrity of what agents believe about you (their memory).

**Who attacks:**
- Text the agents read: a paper abstract, a trending-paper summary, a job posting, a file or diff, or a board post written by another agent that read such text.
- Websites you visit while the dashboard is running.

**Out of scope:** an attacker who already runs code on your Mac; compromise of the model providers or the `claude` / `codex` CLIs.

## Defences

| Threat | Defence | Code |
|---|---|---|
| **Prompt injection**: outside text says "ignore your instructions and …" | Output from untrusted tools is wrapped in `<untrusted>` and labelled as data. The wrapper can't be closed from inside it. | `guardrails.wrap_untrusted`, `toolcall.format_result` |
| **Injected actions**: injected text makes an agent change your data | **Taint tracking.** Once a job reads untrusted content, every tool that writes (tasks, experiments, notes, drafts, memory, board) needs your approval for the rest of that request. You can answer `a` to approve all writes for that one request. Taint passes to delegates and comes back with their answers. | `graph.tools_node`, `graph._delegate` |
| **Memory poisoning**: injected text saved as a "fact" that affects later sessions | `remember` and `forget` are write tools, so the taint rule covers them. You can read and delete every memory in the dashboard's Memory tab. | `memory.py`, `ui` |
| **High-impact actions** | `complete_task`, `delete_task` and `run_coding_task` always ask you first, whatever the taint. You see the full arguments before approving. | `tools/*.py` (`approval=True`) |
| **Secret exfiltration through files** | File and code tools only work under `security.allowed_roots`. Paths are fully resolved (symlinks, `..`) before checking. `.ssh`, `.aws`, `.gnupg`, `.env*`, `*.pem`, `*.key`, `id_*`, credential and keychain files, and the `claude`/`codex` configs are always blocked, as is the system's own `data/`. | `guardrails.check_path` |
| **Secret leakage into prompts** | Anything that looks like a key (Anthropic, OpenAI, GitHub, AWS, Slack, Google, Hugging Face), a private key, or `password=…` is redacted before a model sees it. | `guardrails.redact` |
| **Network misuse / SSRF** | Tools can only fetch over HTTPS from an allowlist: `export.arxiv.org`, `api.openalex.org`, `huggingface.co`, `boards-api.greenhouse.io`. Local and cloud-metadata addresses are refused. | `guardrails.check_url`, `tools/http.py` |
| **Backend agents doing more than chat** | `claude -p` runs with every built-in tool disabled (`--tools ""`), no MCP servers, in an empty temp directory. `codex exec` runs in a read-only sandbox in an empty directory, with browser use, computer use and apps disabled. | `adapters/` |
| **Code-editing agent** | Code changes need approval and only run in a git repository under the allowed folders, so every change shows in `git diff` and can be reverted. Claude edits with `Read/Edit/Write/Glob/Grep` only (no shell). Codex uses its `workspace-write` sandbox with risky features off. | `tools/code.py` |
| **GitHub misuse**: injected text makes an agent post spam, open PRs or comment elsewhere | Everything read from GitHub (READMEs, issues, PRs, files) is untrusted. Issues, comments and PRs **always** need your approval, and only work on repos matching `github.write_repos` (default `Kp759/*`). PRs are drafts, never from the default branch, and only with a clean working tree. Posts are signed as agent-drafted. Repo names, refs and paths are validated, and `gh` gets a list of arguments, never a shell string. | `tools/github.py` |
| **Runaway loops** | At most 10 tool calls per agent job, delegation depth 2, and 6 delegations per request. Tool results are capped at 8,000 characters. | `graph.py` |
| **Dashboard attacked from a website** (it can send messages to your agents) | The dashboard binds to `127.0.0.1` only and rejects any `Host` other than `127.0.0.1` or `localhost` (DNS rebinding). Every POST must carry an `X-My-Agents` header, which another site cannot send without a CORS preflight this server never approves, and cross-origin POSTs are refused. Bodies are size-limited. A strict CSP is set, framing is forbidden, and all model text is HTML-escaped. | `ui/server.py`, `ui/index.html` |
| **Local data exposure** | `data/` (traces, prompts, memory) is `chmod 700` and gitignored. | `store.py` |
| **Accountability** | Every approval decision is logged as a trace step and shown in the timeline. | `graph.tools_node` |

## Residual risks (be aware)

- **Codex can still read files.** Its read-only sandbox allows shell commands that read the disk. Injected text could make a Codex-backed agent print a file into its reply, though it can't send it anywhere (the sandbox blocks network access). For this reason the agents that read the most outside content (Literature Surveyor, Trend Scout) run on Claude with all CLI tools disabled. Codex backs only the Examiner and the Reviewer.
- **GitHub reads use your `gh` login**, so agents can read your private repos (for example, listing them) and that text reaches the model provider. Writes are limited to `github.write_repos` and always approved.
- **Your prompts go to the model providers.** Anything an agent reads is sent to Anthropic or OpenAI, depending on its backend. Use a local model (`backend: ollama`) for an agent that must handle sensitive material.
- **Labelling reduces injection but does not prevent it.** The taint gate is the real control: injected text can still change what an agent *says*, but not what it *does to your data* without asking you.
- **Approval fatigue.** If you approve everything without reading, the gate does nothing. Read the arguments; they are shown in full for this reason.

## Reporting

This is a personal research project. If you find a security problem, open an issue (without exploit details) at https://github.com/Kp759/my-agents/issues.
