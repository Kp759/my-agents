"""Telegram bot: only the owner is served, approvals work by button, the token stays secret."""

import json
import os
import stat
import time

import pytest

from my_agents import telegram as tg
from my_agents.telegram import TelegramBot, load_secrets, save_secrets, split_message, to_telegram_html
from test_chat_ui import call, manager

OWNER, STRANGER = 1111, 2222
TOKEN = "123456789:AAbbccddeeffgghhiijjkkllmmnnooppqq"


class FakeClient:
    def __init__(self):
        self.sent, self.calls = [], []

    def send(self, chat_id, text, buttons=None):
        self.sent.append({"chat_id": chat_id, "text": text, "buttons": buttons})
        return [{"message_id": len(self.sent)}]

    def call(self, method, http_timeout=30, **params):
        self.calls.append((method, params))
        return {"username": "drk_test_bot"} if method == "getMe" else True


def bot(*replies, tmp_path):
    chat, store = manager(*replies)
    client = FakeClient()
    b = TelegramBot(client, OWNER, chat, chat.tracer, store, tmp_path / "tg_state.json")
    return b, client, store


def msg(text, sender=OWNER, chat_type="private", age=0):
    return {"update_id": 1, "message": {"from": {"id": sender}, "chat": {"id": sender, "type": chat_type},
                                        "date": int(time.time()) - age, "text": text}}


def wait_for(cond, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    pytest.fail("timed out")


def test_owner_message_gets_agent_reply(tmp_path):
    b, client, _ = bot("Hello Dr.K", tmp_path=tmp_path)
    b.handle(msg("hi"))
    wait_for(lambda: client.sent)
    assert client.sent[0]["chat_id"] == OWNER
    assert client.sent[0]["text"].startswith("**@task_runner**\nHello Dr.K")


@pytest.mark.parametrize("update", [msg("delete everything", sender=STRANGER), msg("hi", chat_type="group"),
                                    msg("old command", age=3600)])
def test_strangers_groups_and_stale_messages_are_ignored(tmp_path, update):
    b, client, _ = bot(tmp_tool := "should not run", tmp_path=tmp_path)
    b.handle(update)
    time.sleep(0.1)
    assert client.sent == [] and b.chat.status == {}


def test_approval_by_button(tmp_path):
    b, client, store = bot(call("delete_task", id=1), "Deleted.", tmp_path=tmp_path)
    store.insert("tasks", title="old")
    b.handle(msg("delete task 1"))
    wait_for(lambda: client.sent)
    card = client.sent[0]
    assert "delete_task" in card["text"] and card["buttons"]
    data = card["buttons"][0][0]["callback_data"]  # Allow
    # a stranger pressing the button does nothing
    b.handle({"update_id": 2, "callback_query": {"id": "q0", "from": {"id": STRANGER}, "data": data}})
    assert len(store.list("tasks")) == 1
    # a forged / stale request id is refused
    b.handle({"update_id": 3, "callback_query": {"id": "q1", "from": {"id": OWNER}, "data": "ap|deadbeef|yes"}})
    assert len(store.list("tasks")) == 1
    b.handle({"update_id": 4, "callback_query": {"id": "q2", "from": {"id": OWNER}, "data": data,
                                                  "message": {"message_id": 1}}})
    wait_for(lambda: len(client.sent) >= 2)
    assert store.list("tasks") == [] and "Deleted." in client.sent[-1]["text"]
    assert any(m == "editMessageReplyMarkup" for m, _ in client.calls)  # buttons replaced after use


def test_unknown_agent_and_commands(tmp_path):
    b, client, _ = bot(tmp_path=tmp_path)
    b.handle(msg("@nobody hi"))
    assert "Unknown agent" in client.sent[-1]["text"]
    old = b.thread
    b.handle(msg("/new"))
    assert b.thread != old and json.loads((tmp_path / "tg_state.json").read_text())["thread"] == b.thread


def test_html_is_escaped_and_formatted():
    out = to_telegram_html("**Bold** and `code` <script>alert(1)</script>\n```\nx < y\n```")
    assert "<b>Bold</b>" in out and "<code>code</code>" in out
    assert "<script>" not in out and "&lt;script&gt;" in out and "<pre>x &lt; y</pre>" in out


def test_long_messages_are_split():
    parts = split_message("line\n" * 2000)
    assert all(len(p) <= tg.MAX_MESSAGE for p in parts) and len(parts) > 1


def test_token_is_private(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "telegram.env"
    monkeypatch.setenv("MY_AGENTS_TELEGRAM_ENV", str(path))
    save_secrets(TOKEN, OWNER)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    os.chmod(path, 0o644)  # someone loosened it: loading tightens it again
    assert load_secrets() == (TOKEN, OWNER)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert TOKEN not in repr(tg.TelegramClient(TOKEN))


def test_network_errors_do_not_leak_the_token(monkeypatch):
    import urllib.request

    def boom(req, timeout):
        raise OSError(f"failed to reach {req.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(tg.TelegramError) as e:
        tg.TelegramClient(TOKEN).call("getMe")
    assert TOKEN not in str(e.value) and e.value.__cause__ is None


def test_bad_tokens_are_rejected():
    with pytest.raises(tg.TelegramError):
        tg.TelegramClient("not-a-token")
