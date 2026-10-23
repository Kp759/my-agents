"""Talk to your agents from Telegram.

    my-agents telegram setup   create the connection (one time)
    my-agents telegram         run the bot (keep it running)

How it works: the bot long-polls Telegram's Bot API (outbound HTTPS only; nothing
on your machine is exposed to the internet). Each message runs through the same
chat engine as the dashboard. Approvals arrive as Allow / Deny buttons.

Security:
- Only your Telegram user id, in a private chat, is served; everyone else is ignored.
- Messages older than 10 minutes are skipped, so stale commands never run after a restart.
- The bot token lives in a file outside the repo (chmod 600) and is never printed or traced.
- Button presses are checked against the request that is actually waiting.
- Telegram bot chats are not end-to-end encrypted: Telegram's servers can read them.

Study loop: the bot runs the jobs in config.yaml → study.schedule (morning digest, evening
quiz, weekly career check) and sends you the results. A "sticky" job such as the quiz keeps
your replies going to the same agent until you send /done.
"""

from __future__ import annotations

import getpass
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import yaml

from . import study
from .agents import DEFAULT_CONFIG, load_agents
from .chat import ChatManager
from .report import eod_facts, eod_report, format_row, totals
from .store import Store
from .tracing import Tracer

API = "https://api.telegram.org"
TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")
TOKEN_IN_TEXT = re.compile(r"\d{5,}:[A-Za-z0-9_-]{30,}")
MAX_MESSAGE = 4000  # Telegram's limit is 4096 characters
STALE_SECONDS = 600
DEFAULT_SECRETS = "~/.config/my-agents/telegram.env"


class TelegramError(RuntimeError):
    pass


# ---- configuration and secrets --------------------------------------------------------


def _config() -> dict:
    data = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    return {"telegram": data.get("telegram") or {}, "owner": data.get("owner") or "", "eod": data.get("eod")}


def _full_config() -> dict:  # re-read every time, so schedule edits apply without a restart
    return yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}


def secrets_path() -> Path:
    return Path(os.environ.get("MY_AGENTS_TELEGRAM_ENV") or _config()["telegram"].get("secrets_file")
                or DEFAULT_SECRETS).expanduser()


def load_secrets() -> tuple[str, int]:
    path = secrets_path()
    if not path.exists():
        raise TelegramError(f"not set up yet: run `my-agents telegram setup` (expected {path})")
    if path.stat().st_mode & 0o077:
        os.chmod(path, 0o600)  # the token must only be readable by you
    values = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            values[key.strip()] = value.strip()
    token, user = values.get("TELEGRAM_BOT_TOKEN", ""), values.get("TELEGRAM_USER_ID", "")
    if not TOKEN_RE.match(token) or not user.isdigit():
        raise TelegramError(f"{path} is incomplete: run `my-agents telegram setup` again")
    return token, int(user)


def extract_token(pasted: str) -> str | None:
    """Find a bot token in what was pasted: the token alone, or the whole BotFather message."""
    found = TOKEN_IN_TEXT.findall(pasted)
    return found[0] if len(found) == 1 else None


def _describe_paste(pasted: str) -> str:
    # Describe the problem without echoing anything that might be a secret.
    if not pasted.strip():
        return "nothing was pasted (the prompt stays blank while you paste; that's normal)"
    if ":" not in pasted:
        return f"got {len(pasted.strip())} characters with no ':'; a token looks like 1234567890:AAE…"
    if len(TOKEN_IN_TEXT.findall(pasted)) > 1:
        return "found more than one token; paste just the one from BotFather"
    return f"got {len(pasted.strip())} characters but no valid token; copy the whole token line from BotFather"


def save_secrets(token: str, user_id: int) -> Path:
    path = secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("# Telegram bot for my-agents. Keep this file private; never commit it.\n"
                f"TELEGRAM_BOT_TOKEN={token}\nTELEGRAM_USER_ID={user_id}\n")
    return path


# ---- Bot API client -------------------------------------------------------------------


class TelegramClient:
    def __init__(self, token: str):
        if not TOKEN_RE.match(token):
            raise TelegramError("that does not look like a bot token (expected 123456:ABC…)")
        self._token = token

    def __repr__(self) -> str:  # never show the token in logs or tracebacks
        return "TelegramClient(token=***)"

    def call(self, method: str, http_timeout: int = 30, **params):
        """Call a Bot API method. `http_timeout` is ours; Telegram params (e.g. `timeout`) go in params."""
        req = urllib.request.Request(
            f"{API}/bot{self._token}/{method}", data=json.dumps(params).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                data = json.loads(e.read())
            except (json.JSONDecodeError, OSError):
                raise TelegramError(f"{method}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise TelegramError(f"{method}: network error ({type(e).__name__})") from None
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('description', 'failed')}")
        return data["result"]

    def send(self, chat_id: int, text: str, buttons: list[list[dict]] | None = None) -> list[dict]:
        """Send markdown-ish text as Telegram HTML, split to fit; falls back to plain text."""
        sent = []
        chunks = split_message(text)
        for i, chunk in enumerate(chunks):
            extra = {"reply_markup": {"inline_keyboard": buttons}} if buttons and i == len(chunks) - 1 else {}
            try:
                sent.append(self.call("sendMessage", chat_id=chat_id, text=to_telegram_html(chunk),
                                      parse_mode="HTML", disable_web_page_preview=True, **extra))
            except TelegramError:
                sent.append(self.call("sendMessage", chat_id=chat_id, text=chunk,
                                      disable_web_page_preview=True, **extra))
        return sent


def split_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    text = text.strip() or "(empty reply)"
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > limit // 2 else limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks + [text]


def to_telegram_html(text: str) -> str:
    """Convert the agents' markdown to the small HTML subset Telegram accepts."""
    out = []
    for i, part in enumerate(re.split(r"```[^\n]*\n?", text)):
        if i % 2:  # inside a fenced code block
            out.append(f"<pre>{html.escape(part.rstrip())}</pre>")
            continue
        s = html.escape(part)
        s = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", s, flags=re.M)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
        s = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)
        s = re.sub(r"^(\s*)[-*] ", r"\1• ", s, flags=re.M)
        out.append(s)
    return "".join(out)


# ---- the bot ----------------------------------------------------------------------------

HELP = """<b>{team}</b>
Just write a message: the router picks the right agent.
<code>@writer …</code> sends it to one agent.

/new: start a fresh conversation
/status: what's running now
/trace: the steps of the last request
/agents: who's on the team
/eod: today's end-of-day report
/jobs: your study schedule · /run NAME: run a job now
/due: topics due for review · /done: leave quiz mode
/help: this message"""


class TelegramBot:
    def __init__(self, client: TelegramClient, user_id: int, chat: ChatManager, tracer: Tracer, store: Store,
                 state_file: Path):
        self.client, self.user_id, self.chat, self.tracer, self.store = client, user_id, chat, tracer, store
        self.state_file = state_file
        self.state = json.loads(state_file.read_text()) if state_file.exists() else {}
        self.owner = _config()["owner"]
        self.team = f"{self.owner}'s agents" if self.owner else "Your agents"

    # -- state ---------------------------------------------------------------------------

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(self.state))

    @property
    def thread(self) -> str:
        if not self.state.get("thread"):
            self.state["thread"] = f"tg-{self.chat.new_thread()}"
            self._save()
        return self.state["thread"]

    # -- polling -------------------------------------------------------------------------

    def run(self) -> None:
        me = self.client.call("getMe")
        print(f"Telegram bot @{me['username']} is running for user {self.user_id}. Ctrl+C to stop.")
        offset, backoff = None, 1
        while True:
            try:
                # long poll: Telegram holds the request up to 50 s until a message arrives
                updates = self.client.call("getUpdates", http_timeout=60, offset=offset, timeout=50,
                                           allowed_updates=["message", "callback_query"])
                backoff = 1
            except TelegramError as e:
                print(f"telegram: {e}; retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    self.handle(update)
                except Exception as e:  # one bad update must not stop the bot
                    print(f"telegram: error handling update: {type(e).__name__}: {e}")
            self.maybe_send_eod()
            self.maybe_run_jobs()

    def handle(self, update: dict) -> None:
        if "callback_query" in update:
            return self.on_button(update["callback_query"])
        msg = update.get("message") or {}
        sender = (msg.get("from") or {}).get("id")
        if sender != self.user_id or (msg.get("chat") or {}).get("type") != "private":
            print(f"telegram: ignored a message from user {sender} (not the owner)")
            return
        if time.time() - msg.get("date", 0) > STALE_SECONDS:
            return  # sent while the bot was offline; don't act on old commands
        text = (msg.get("text") or "").strip()
        if not text:
            self.reply("I can only read text messages.")
            return
        if text.startswith("/"):
            head, _, arg = text.partition(" ")
            return self.command(head.split("@")[0].lower(), arg)
        self.ask(text)

    # -- commands ------------------------------------------------------------------------

    def reply(self, text: str, buttons=None) -> list[dict]:
        return self.client.send(self.user_id, text, buttons)

    def command(self, cmd: str, arg: str = "") -> None:
        if cmd in ("/start", "/help"):
            self.client.call("sendMessage", chat_id=self.user_id, text=HELP.format(team=html.escape(self.team)),
                             parse_mode="HTML")
        elif cmd == "/new":
            self.state["thread"] = f"tg-{self.chat.new_thread()}"
            self.state.pop("sticky", None)
            self._save()
            self.reply("Started a fresh conversation.")
        elif cmd == "/status":
            st = self.chat.status.get(self.thread, {"state": "idle"})
            self.reply(f"Conversation `{self.thread}`: {st['state']}.")
        elif cmd == "/agents":
            self.reply("\n".join(f"**@{s.name}** ({s.backend} {s.model or ''}): {s.description}"
                                 for s in load_agents().values()))
        elif cmd == "/trace":
            rid = self.chat.status.get(self.thread, {}).get("request_id") or self.tracer.last_run_id()
            rows = self.tracer.run(rid) if rid else []
            if not rows:
                self.reply("No request yet.")
            else:
                lines = [format_row(r) for r in rows]
                self.reply("```\n" + "\n".join(lines)[-3500:] + "\n```\n" + _footer(totals(rows)))
        elif cmd == "/eod":
            self.reply(self._eod_text())
        elif cmd == "/jobs":
            js = study.jobs(_full_config())
            self.reply("\n".join(f"**{j['name']}**: {j['at']} {','.join(j.get('days') or ['daily'])} → @{j['agent']}"
                                  for j in js) or "No jobs in config.yaml → study.schedule.")
        elif cmd == "/run":
            js = {j["name"]: j for j in study.jobs(_full_config())}
            name = arg.strip()
            if name not in js:
                self.reply(f"Run which job? {', '.join(js) or 'none configured'}")
            else:
                self.run_job(js[name])
        elif cmd == "/due":
            d = study.due(self.store)
            lines = [f"• {r['topic']} (due {r['due']}, last score {r['last_score'] if r['last_score'] is not None else '–'})"
                     for r in d["due"]]
            self.reply("**Due for review today**\n" + ("\n".join(lines) or "Nothing due. 🎉") +
                       (f"\n\nNext: {d['upcoming'][0]['topic']} on {d['upcoming'][0]['due']}" if d["upcoming"] else ""))
        elif cmd == "/done":
            self.state.pop("sticky", None)
            self._save()
            self.reply("Left quiz mode: the router picks agents again.")
        else:
            self.reply("Unknown command. Try /help.")

    # -- asking agents ---------------------------------------------------------------------

    def ask(self, text: str) -> None:
        agent = None
        sticky = self.state.get("sticky") or {}
        if sticky and time.time() < sticky.get("until", 0) and not text.startswith("@"):
            agent = sticky["agent"]  # answering a quiz: keep talking to the examiner
        if text.startswith("@"):
            head, _, rest = text.partition(" ")
            agent, text = head[1:].lower(), rest.strip()
            if agent not in self.chat.agents:
                self.reply(f"Unknown agent @{agent}. Send /agents to see the team.")
                return
        before = len(self.chat.state(self.thread)["messages"])
        try:
            self.chat.send(self.thread, text, agent)
        except ValueError as e:
            self.reply(str(e))
            return
        threading.Thread(target=self._watch, args=(before,), daemon=True).start()

    def _watch(self, before: int) -> None:
        try:
            self._watch_inner(before)
        except Exception as e:  # never fail silently: the user is waiting on their phone
            print(f"telegram: error while waiting for agents: {type(e).__name__}: {e}")
            try:
                self.reply(f"⚠️ I lost track of that request ({type(e).__name__}). Check the dashboard or /trace.")
            except TelegramError:
                pass

    def _watch_inner(self, before: int) -> None:
        """Wait for the request to finish or pause, then report back."""
        thread, last_typing = self.thread, 0.0
        while True:
            st = self.chat.state(thread)
            if st["state"] != "running":
                break
            if time.time() - last_typing > 4:
                try:
                    self.client.call("sendChatAction", chat_id=self.user_id, action="typing")
                except TelegramError:
                    pass
                last_typing = time.time()
            time.sleep(1)
        if st["state"] == "approval":
            a = st["approval"]
            rid = st["request_id"]
            args = "\n".join(f"{k}: {v}" for k, v in a["args"].items())
            buttons = [[{"text": "✅ Allow", "callback_data": f"ap|{rid[:16]}|yes"},
                        {"text": "❌ Deny", "callback_data": f"ap|{rid[:16]}|no"}]]
            if a.get("allow_all"):
                buttons.append([{"text": "Allow all for this request", "callback_data": f"ap|{rid[:16]}|all"}])
            self.reply(f"⚠️ **@{a['agent']}** wants to run `{a['tool']}`\n_{a.get('reason', '')}_\n\n"
                       f"```\n{args}\n```\nRead it before allowing.", buttons)
            self.state["waiting"] = before
            self._save()
            return
        if st["state"] == "error":
            self.reply(f"⚠️ Something went wrong: {st.get('error')}")
            return
        new = [m for m in st["messages"][before:] if m["role"] == "assistant"]
        rows = self.tracer.run(st["request_id"])
        for m in new:
            self.reply(f"**@{m.get('agent', 'agent')}**\n{m['content']}\n\n" + _footer(totals(rows) if rows else None))

    def on_button(self, q: dict) -> None:
        if (q.get("from") or {}).get("id") != self.user_id:
            return
        kind, _, rest = (q.get("data") or "").partition("|")
        rid, _, decision = rest.partition("|")
        st = self.chat.status.get(self.thread, {})
        ok = (kind == "ap" and decision in ("yes", "no", "all") and st.get("state") == "approval"
              and st.get("request_id", "").startswith(rid))
        if not ok:
            self.client.call("answerCallbackQuery", callback_query_id=q["id"], text="This request is no longer waiting.")
            return
        self.chat.decide(self.thread, decision)
        label = {"yes": "✅ Allowed", "no": "❌ Denied", "all": "✅ Allowed everything for this request"}[decision]
        self.client.call("answerCallbackQuery", callback_query_id=q["id"], text=label)
        msg = q.get("message") or {}
        try:  # replace the buttons with the decision, so it can't be pressed twice
            self.client.call("editMessageReplyMarkup", chat_id=self.user_id, message_id=msg.get("message_id"),
                             reply_markup={"inline_keyboard": [[{"text": label, "callback_data": "done"}]]})
        except TelegramError:
            pass
        threading.Thread(target=self._watch, args=(self.state.get("waiting", 0),), daemon=True).start()

    # -- end of day ------------------------------------------------------------------------

    def _eod_text(self) -> str:
        cfg = _config()
        facts = eod_facts(self.tracer, self.store, date.today())
        return eod_report(facts, cfg["eod"], owner=self.owner or "the user")

    # -- study loop --------------------------------------------------------------------------

    def run_job(self, job: dict) -> bool:
        """Send a scheduled job's prompt to its agent in the current conversation."""
        if self.chat.status.get(self.thread, {}).get("state") in ("running", "approval"):
            return False  # busy; the scheduler tries again on the next poll
        self.reply(f"⏰ **{job['name']}** → @{job['agent']}")
        if job.get("sticky"):
            self.state["sticky"] = {"agent": job["agent"], "until": time.time() + 3 * 3600}
            self._save()
            self.reply(f"Your replies now go to @{job['agent']}. Send /done to stop.")
        self.ask(f"@{job['agent']} {job['prompt']}")
        return True

    def maybe_run_jobs(self) -> None:
        ran = self.state.setdefault("jobs_ran", {})
        for job in study.jobs(_full_config()):
            if study.is_due(job, datetime.now(), ran.get(job["name"])) and self.run_job(job):
                ran[job["name"]] = date.today().isoformat()
                self._save()
                return  # one job at a time; the next one runs once this conversation is free

    def maybe_send_eod(self) -> None:
        at = _config()["telegram"].get("eod_time")
        if not at:
            return
        today = date.today().isoformat()
        hour, _, minute = str(at).partition(":")
        due = datetime.now().replace(hour=int(hour), minute=int(minute or 0), second=0, microsecond=0)
        if datetime.now() >= due and self.state.get("eod_sent") != today:
            self.state["eod_sent"] = today
            self._save()
            threading.Thread(target=lambda: self.reply(self._eod_text()), daemon=True).start()


def _footer(t: dict | None) -> str:
    if not t:
        return ""
    money = f"${t['cost_usd']:.4f}" + (" + tokens" if t["unpriced_backends"] else "")
    return (f"_{t['wall_s']:.0f}s · {t['model_calls']} model calls · {t['tool_calls']} tools · "
            f"{t['delegations']} delegations · {money} API-equivalent_")


# ---- entry points -----------------------------------------------------------------------


def run() -> None:
    token, user_id = load_secrets()
    store, tracer = Store(), Tracer()
    bot = TelegramBot(TelegramClient(token), user_id, ChatManager(store, tracer), tracer, store,
                      Path(store.path).with_name("telegram_state.json"))
    try:
        bot.run()
    except KeyboardInterrupt:
        print()


def setup() -> None:
    print("""Connect Telegram to your agents (about 2 minutes).

1. In Telegram, open @BotFather and send /newbot.
2. Choose a name and a username ending in "bot".
3. BotFather replies with a token like 123456789:AA…. Copy it.
""")
    for attempt in range(3):
        pasted = getpass.getpass("Paste the bot token here (input is hidden, nothing will show), then Enter: ")
        token = extract_token(pasted)
        if not token:
            print(f"  ✗ {_describe_paste(pasted)}. Try again.")
            continue
        client = TelegramClient(token)
        try:
            me = client.call("getMe")
            break
        except TelegramError as e:
            print(f"  ✗ Telegram rejected that token ({e}). If you revoked it, use the new one. Try again.")
    else:
        raise TelegramError("no working token after 3 tries; run `my-agents telegram setup` again")
    print(f"\n✓ Token works: your bot is @{me['username']}.")
    print(f"\n4. Now open https://t.me/{me['username']} and press Start (or send /start).")
    print("   Waiting up to 3 minutes for your message…")
    deadline, offset = time.time() + 180, None
    while time.time() < deadline:
        updates = client.call("getUpdates", http_timeout=40, offset=offset, timeout=30)
        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message") or {}
            sender = msg.get("from") or {}
            if (msg.get("chat") or {}).get("type") != "private" or not sender.get("id"):
                continue
            name = " ".join(filter(None, [sender.get("first_name"), sender.get("last_name")]))
            handle = f" (@{sender['username']})" if sender.get("username") else ""
            if input(f"\nGot a message from {name}{handle}, user id {sender['id']}. Is this you? [y/N] ").lower() != "y":
                continue
            path = save_secrets(token, sender["id"])
            client.call("getUpdates", offset=offset, timeout=0)  # mark setup messages as read
            client.call("sendMessage", chat_id=sender["id"],
                        text="✅ Connected. Run `my-agents telegram` on your Mac, then message me here.")
            print(f"\n✓ Saved to {path} (readable only by you).\n  Only user {sender['id']} will be answered.")
            print("  Start the bot with:  my-agents telegram")
            return
    raise TelegramError("no message received; run setup again and press Start in Telegram")
