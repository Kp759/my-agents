"""Your calendar, read-only, from iCalendar (.ics) feeds. No API keys, no OAuth.

Every calendar app can publish a private iCal address: Google Calendar (Settings → your
calendar → "Secret address in iCal format"), iCloud (share → Public Calendar), Outlook
(Settings → Shared calendars → Publish). Add one with `my-agents calendar add NAME`; you
can also add a local .ics file. The addresses are secrets (anyone with one can read the
calendar), so they live in ~/.config/my-agents/calendars.txt (chmod 600, outside the repo),
are typed with hidden input, and never appear in prompts, traces, tool results or errors.

calendar_events: events in a date range, recurring events expanded, optional text filter.
free_slots:      the free gaps in one day's working hours.

Both are read-only. Their results are untrusted: anyone can send you an invite, and its
title or description can contain prompt injection (a known attack on calendar assistants).

The parser below is deliberately small and readable instead of a library: line unfolding,
properties with parameters, time zones (IANA names, a few Windows names, UTC, floating),
all-day events, DURATION, RRULE (DAILY/WEEKLY/MONTHLY/YEARLY with INTERVAL, COUNT, UNTIL,
BYDAY incl. "2TU"/"-1FR", BYMONTHDAY, BYMONTH, BYSETPOS, WKST), RDATE, EXDATE, moved or
cancelled instances (RECURRENCE-ID) and cancelled events.
"""

from __future__ import annotations

import calendar as _cal
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import Tool

DEFAULT_SOURCES = "~/.config/my-agents/calendars.txt"
MAX_BYTES = 20_000_000  # a calendar with years of history can be large
CACHE_SECONDS = 300
MAX_DAYS = 62  # widest range one call may ask for
NOTES_CHARS = 200
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
WEEKDAYS = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
WINDOWS_ZONES = {  # Outlook writes Windows zone names
    "Eastern Standard Time": "America/New_York", "Central Standard Time": "America/Chicago",
    "Mountain Standard Time": "America/Denver", "Pacific Standard Time": "America/Los_Angeles",
    "GMT Standard Time": "Europe/London", "W. Europe Standard Time": "Europe/Berlin",
    "Romance Standard Time": "Europe/Paris", "Central Europe Standard Time": "Europe/Budapest",
    "India Standard Time": "Asia/Kolkata", "China Standard Time": "Asia/Shanghai",
    "Tokyo Standard Time": "Asia/Tokyo", "AUS Eastern Standard Time": "Australia/Sydney",
    "UTC": "UTC", "Coordinated Universal Time": "UTC",
}

_cache: dict[str, tuple[float, str]] = {}


# ---- sources (secrets) ----------------------------------------------------------------


def sources_path() -> Path:
    return Path(os.environ.get("MY_AGENTS_CALENDARS") or DEFAULT_SOURCES).expanduser()


def load_sources() -> dict[str, str]:
    """name -> https URL or .ics path."""
    path = sources_path()
    if not path.exists():
        return {}
    if path.stat().st_mode & 0o077:
        os.chmod(path, 0o600)
    out = {}
    for line in path.read_text().splitlines():
        name, sep, source = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            out[name.strip()] = source.strip()
    return out


def _write_sources(sources: dict[str, str]) -> None:
    path = sources_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("# Calendars for my-agents (name = iCal address or .ics path). Private; never commit.\n")
        f.writelines(f"{n} = {s}\n" for n, s in sources.items())


def normalise_source(source: str) -> str:
    source = source.strip()
    if source.lower().startswith("webcal://"):
        source = "https://" + source[len("webcal://"):]
    if source.lower().startswith("http://"):
        raise ValueError("use the https:// address (calendar addresses are secrets; never send them unencrypted)")
    return source


def save_source(name: str, source: str) -> None:
    if not NAME_RE.match(name):
        raise ValueError("calendar names are 1-40 letters, digits, _ or -")
    _write_sources({**load_sources(), name: normalise_source(source)})


def remove_source(name: str) -> bool:
    sources = load_sources()
    if name not in sources:
        return False
    del sources[name]
    _write_sources(sources)
    return True


def _fetch(source: str) -> str:
    hit = _cache.get(source)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    if source.startswith("https://"):
        from .web import _download  # pinned public IP, https only, redirects rechecked

        _, _, body = _download(source, timeout=30, max_bytes=MAX_BYTES, accept="text/calendar,*/*;q=0.1")
        text = body.decode("utf-8", "replace")
    else:
        text = Path(source).expanduser().read_text(encoding="utf-8", errors="replace")
    if not text.lstrip("﻿ \r\n").startswith("BEGIN:VCALENDAR"):
        raise ValueError("not an iCalendar (.ics) feed")
    _cache[source] = (time.time(), text)
    return text


def _safe_error(e: Exception, source: str) -> str:
    """An error message that can't leak the secret address (HTTP errors include the URL path)."""
    from .http import HTTPStatusError

    if isinstance(e, HTTPStatusError):
        return f"HTTP {e.status}: the calendar address may have been reset; add it again"
    if source.startswith("https://"):
        return f"{type(e).__name__} while downloading the feed"
    return f"{type(e).__name__}: {e}".replace(source, "<file>")


# ---- time zones ------------------------------------------------------------------------


def local_zone():
    """The user's IANA time zone (DST-aware), not just today's UTC offset."""
    for name in (os.environ.get("TZ"), _zone_from_localtime()):
        if name:
            try:
                return ZoneInfo(name.lstrip(":"))
            except (ZoneInfoNotFoundError, ValueError):
                pass
    return datetime.now().astimezone().tzinfo


def _zone_from_localtime() -> str | None:
    try:
        target = os.readlink("/etc/localtime")
    except OSError:
        return None
    return target.split("zoneinfo/", 1)[1] if "zoneinfo/" in target else None


def _zone(tzid: str | None, default):
    if not tzid:
        return default
    tzid = tzid.strip('"')
    for name in (tzid, WINDOWS_ZONES.get(tzid), tzid.split("/", 1)[-1] if tzid.startswith("/") else None):
        if name:
            try:
                return ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                pass
    return default


# ---- parsing ---------------------------------------------------------------------------


def _unfold(text: str) -> list[str]:
    """Join continuation lines (a line starting with a space or tab continues the previous one)."""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        elif raw:
            lines.append(raw)
    return lines


def _split(line: str) -> tuple[str, dict[str, str], str]:
    """'DTSTART;TZID=Europe/Berlin:20261023T090000' -> ('DTSTART', {'TZID': 'Europe/Berlin'}, '2026…')."""
    head, value, in_quotes = line, "", False
    for i, ch in enumerate(line):
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == ":" and not in_quotes:
            head, value = line[:i], line[i + 1:]
            break
    name, *params = head.split(";")
    return name.upper(), {k.upper(): v for k, _, v in (p.partition("=") for p in params)}, value


def _text(value: str) -> str:
    return re.sub(r"\\([\\,;nN])", lambda m: "\n" if m.group(1) in "nN" else m.group(1), value).strip()


def _when(value: str, params: dict, default_tz) -> tuple[datetime | date, bool]:
    """A DATE (all day) or DATE-TIME, as (date, True) or (aware datetime, False)."""
    value = value.strip()
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        return datetime.strptime(value[:8], "%Y%m%d").date(), True
    if value.endswith("Z"):
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc), False
    return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_zone(params.get("TZID"), default_tz)), False


def _duration(value: str) -> timedelta:
    m = re.fullmatch(r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", value.strip())
    if not m:
        return timedelta(0)
    sign, w, d, h, mi, s = m.groups()
    delta = timedelta(weeks=int(w or 0), days=int(d or 0), hours=int(h or 0), minutes=int(mi or 0), seconds=int(s or 0))
    return -delta if sign == "-" else delta


def parse_ics(text: str, calendar: str, tz) -> list[dict]:
    """The VEVENTs of one feed, unexpanded."""
    events, ev, depth = [], None, 0
    for line in _unfold(text):
        name, params, value = _split(line)
        if name == "BEGIN" and value.upper() == "VEVENT":
            ev, depth = {"calendar": calendar, "exdates": [], "rdates": []}, 0
        elif ev is None:
            continue
        elif name == "BEGIN":  # a VALARM inside the event
            depth += 1
        elif name == "END" and depth:
            depth -= 1
        elif name == "END" and value.upper() == "VEVENT":
            if "start" in ev:
                events.append(ev)
            ev = None
        elif depth:
            continue
        elif name == "DTSTART":
            ev["start"], ev["all_day"] = _when(value, params, tz)
        elif name == "DTEND":
            ev["end"], _ = _when(value, params, tz)
        elif name == "DURATION":
            ev["duration"] = _duration(value)
        elif name in ("SUMMARY", "LOCATION", "DESCRIPTION", "UID", "STATUS", "TRANSP"):
            ev[name.lower()] = _text(value)
        elif name == "RRULE":
            ev["rrule"] = {k.upper(): v for k, _, v in (p.partition("=") for p in value.split(";"))}
        elif name in ("EXDATE", "RDATE"):
            ev["exdates" if name == "EXDATE" else "rdates"] += [_when(v, params, tz)[0] for v in value.split(",") if v]
        elif name == "RECURRENCE-ID":
            ev["recurrence_id"], _ = _when(value, params, tz)
    return events


# ---- recurrence ------------------------------------------------------------------------


def _byday(token: str) -> tuple[int | None, int]:
    m = re.fullmatch(r"([+-]?\d{1,2})?(MO|TU|WE|TH|FR|SA|SU)", token.strip().upper())
    if not m:
        raise ValueError(f"bad BYDAY {token!r}")
    return (int(m.group(1)) if m.group(1) else None), WEEKDAYS.index(m.group(2))


def _day_matches(rule: dict, d0: date, d: date) -> bool:
    """Does day d satisfy the rule's period and BY* filters (ignoring COUNT, UNTIL, BYSETPOS)?"""
    freq, interval = rule.get("FREQ", "DAILY"), int(rule.get("INTERVAL", 1) or 1)
    if freq == "DAILY":
        if (d - d0).days % interval:
            return False
    elif freq == "WEEKLY":
        wkst = WEEKDAYS.index(rule.get("WKST", "MO"))
        week_start = lambda x: x - timedelta(days=(x.weekday() - wkst) % 7)  # noqa: E731
        if ((week_start(d) - week_start(d0)).days // 7) % interval:
            return False
    elif freq == "MONTHLY":
        if ((d.year - d0.year) * 12 + d.month - d0.month) % interval:
            return False
    elif freq == "YEARLY":
        if (d.year - d0.year) % interval:
            return False
    else:  # HOURLY and finer are not calendar events people ask about
        return d == d0
    months = [int(m) for m in rule["BYMONTH"].split(",")] if rule.get("BYMONTH") else None
    if months and d.month not in months:
        return False
    if freq == "YEARLY" and not months and not rule.get("BYDAY") and not rule.get("BYMONTHDAY") and d.month != d0.month:
        return False
    dim = _cal.monthrange(d.year, d.month)[1]
    if rule.get("BYMONTHDAY"):
        days = {int(x) if int(x) > 0 else dim + int(x) + 1 for x in rule["BYMONTHDAY"].split(",")}
        if d.day not in days:
            return False
    if rule.get("BYDAY"):
        ok = False
        for n, wd in map(_byday, rule["BYDAY"].split(",")):
            if d.weekday() != wd:
                continue
            if n is None or freq in ("DAILY", "WEEKLY"):
                ok = True
            elif n > 0 and (d.day - 1) // 7 + 1 == n:
                ok = True
            elif n < 0 and (dim - d.day) // 7 + 1 == -n:
                ok = True
        if not ok:
            return False
    elif not rule.get("BYMONTHDAY"):  # no BY-day rule: repeat on the start's weekday / day of month
        if freq == "WEEKLY" and d.weekday() != d0.weekday():
            return False
        if freq in ("MONTHLY", "YEARLY") and d.day != d0.day:
            return False
    return True


def _rule_days(rule: dict, d0: date, last: date):
    """Days from d0 to `last` (inclusive) on which the rule fires, the start day first."""
    yield d0
    setpos = [int(x) for x in rule["BYSETPOS"].split(",")] if rule.get("BYSETPOS") else None
    d = d0 + timedelta(days=1)
    while d <= last:
        if _day_matches(rule, d0, d):
            if not setpos:
                yield d
            else:  # e.g. "last weekday of the month": positions among the month's matching days
                dim = _cal.monthrange(d.year, d.month)[1]
                month = [x for x in (date(d.year, d.month, i) for i in range(1, dim + 1)) if _day_matches(rule, d0, x)]
                if any(month[p - 1 if p > 0 else p] == d for p in setpos if -len(month) <= p <= len(month) and p):
                    yield d
        d += timedelta(days=1)


def _occurrences(ev: dict, until: datetime, tz) -> list[datetime | date]:
    """Start times of an event up to `until`."""
    start, rule = ev["start"], ev.get("rrule")
    if not rule:
        return [start]
    all_day = ev["all_day"]
    d0 = start if all_day else start.date()
    limit = until.date() if all_day else until.astimezone(start.tzinfo).date()
    if rule.get("UNTIL"):
        u, u_all_day = _when(rule["UNTIL"], {}, start.tzinfo if not all_day else tz)
        u_date = u if u_all_day else (u.astimezone(start.tzinfo).date() if not all_day else u.date())
        limit = min(limit, u_date)
    count = int(rule["COUNT"]) if rule.get("COUNT") else None
    out = []
    for i, d in enumerate(_rule_days(rule, d0, limit)):
        if count is not None and i >= count:
            break
        occ = d if all_day else datetime.combine(d, start.timetz().replace(tzinfo=None), tzinfo=start.tzinfo)
        if rule.get("UNTIL") and not all_day and not u_all_day and occ > u:
            break
        out.append(occ)
    return out


def _as_local(x: datetime | date, tz) -> datetime:
    if isinstance(x, datetime):
        return x.astimezone(tz)
    return datetime.combine(x, datetime.min.time(), tzinfo=tz)


def _day_of(x: datetime | date) -> date:
    return x.date() if isinstance(x, datetime) else x


def expand(events: list[dict], start: datetime, end: datetime, tz) -> list[dict]:
    """Event instances that overlap [start, end), in the user's time zone, sorted by start."""
    moved = {(e.get("uid"), _as_local(e["recurrence_id"], tz)) for e in events if e.get("recurrence_id")}
    out = []
    for ev in events:
        if ev.get("status", "").upper() == "CANCELLED":
            continue
        s0 = ev["start"]
        if ev["all_day"]:  # count whole days, so a DST change doesn't stretch an all-day event past midnight
            ndays = (_day_of(ev["end"]) - s0).days if "end" in ev else (ev.get("duration") or timedelta(days=1)).days
            length = timedelta(days=max(1, ndays))
        elif "end" in ev:
            length = _as_local(ev["end"], tz) - _as_local(s0, tz)
        else:
            length = ev.get("duration") or timedelta(0)
        excluded = {_as_local(x, tz) for x in ev["exdates"]}
        starts = _occurrences(ev, end, tz) if not ev.get("recurrence_id") else [s0]
        for occ in [*starts, *ev["rdates"]]:
            s = _as_local(occ, tz)
            if s in excluded or (not ev.get("recurrence_id") and ev.get("rrule") and (ev.get("uid"), s) in moved):
                continue
            e = _as_local(_day_of(occ) + length, tz) if ev["all_day"] else s + length
            if s < end and (e > start or (e == s and start <= s)):
                out.append({**ev, "s": s, "e": e})
    out.sort(key=lambda x: (x["s"], x.get("summary", "")))
    return out


# ---- the tools -------------------------------------------------------------------------


def _day(value: str | None, tz) -> date:
    today = datetime.now(tz).date()
    v = (value or "today").strip().lower()
    if v in ("today", "now"):
        return today
    if v in ("tomorrow", "yesterday"):
        return today + timedelta(days=1 if v == "tomorrow" else -1)
    names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if v in names:
        return today + timedelta(days=(names.index(v) - today.weekday()) % 7)
    try:
        return date.fromisoformat(v)
    except ValueError:
        raise ValueError(f"bad date {value!r}: use YYYY-MM-DD, today, tomorrow or a weekday name") from None


def _load(start: datetime, end: datetime, tz) -> tuple[list[dict], list[dict]]:
    sources = load_sources()
    if not sources:
        raise LookupError("no calendar connected")
    events, problems = [], []
    for name, source in sources.items():
        try:
            events += expand(parse_ics(_fetch(source), name, tz), start, end, tz)
        except Exception as e:  # one broken feed shouldn't hide the others
            problems.append({"calendar": name, "error": _safe_error(e, source)})
    events.sort(key=lambda x: (x["s"], x.get("summary", "")))
    return events, problems


NOT_CONNECTED = {"note": "No calendar is connected yet. The user can connect one by running "
                         "`my-agents calendar add NAME` in a terminal (Google, iCloud and Outlook all "
                         "publish a private iCal address). Tasks and milestones are in list_tasks and list_milestones."}


def _show(ev: dict) -> dict:
    s, e = ev["s"], ev["e"]
    if ev["all_day"]:
        days = max(1, (e.date() - s.date()).days)
        when = s.strftime("%a %d %b") + (f" – {(e - timedelta(days=1)).strftime('%a %d %b')}" if days > 1 else "") + ", all day"
    else:
        when = f"{s.strftime('%a %d %b %H:%M')}–{e.strftime('%H:%M') if e.date() == s.date() else e.strftime('%a %d %b %H:%M')}"
    out = {"when": when, "title": ev.get("summary") or "(no title)", "calendar": ev["calendar"]}
    if ev.get("location"):
        out["location"] = ev["location"][:120]
    if ev.get("description"):
        notes = ev["description"]
        out["notes"] = notes[:NOTES_CHARS] + ("…" if len(notes) > NOTES_CHARS else "")
    if ev.get("rrule") or ev.get("recurrence_id"):
        out["recurring"] = True
    if (ev.get("transp") or "").upper() == "TRANSPARENT":
        out["shows_as"] = "free"
    return out


def calendar_events(start: str | None = None, days: int = 7, query: str | None = None, limit: int = 50) -> dict:
    tz = local_zone()
    days = max(1, min(int(days), MAX_DAYS))
    s = datetime.combine(_day(start, tz), datetime.min.time(), tzinfo=tz)
    e = s + timedelta(days=days)
    try:
        events, problems = _load(s, e, tz)
    except LookupError:
        return NOT_CONNECTED
    if query:
        q = query.lower()
        events = [x for x in events if q in " ".join(str(x.get(k, "")) for k in ("summary", "location", "description")).lower()]
    limit = max(1, min(int(limit), 100))
    return {
        "range": f"{s.date()} to {(e - timedelta(days=1)).date()} ({tz})",
        "events": [_show(x) for x in events[:limit]],
        "total": len(events),
        **({"more": f"{len(events) - limit} more; narrow the range or use query"} if len(events) > limit else {}),
        **({"problems": problems} if problems else {}),
    }


def _clock(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", value.strip())
    if not m or int(m.group(1)) > 24 or int(m.group(2)) > 59:
        raise ValueError(f"bad time {value!r}: use HH:MM")
    return int(m.group(1)), int(m.group(2))


def free_slots(date: str | None = None, day_start: str = "09:00", day_end: str = "18:00", min_minutes: int = 30) -> dict:
    tz = local_zone()
    d = _day(date, tz)
    midnight = datetime.combine(d, datetime.min.time(), tzinfo=tz)
    (h0, m0), (h1, m1) = _clock(day_start), _clock(day_end)
    lo, hi = midnight + timedelta(hours=h0, minutes=m0), midnight + timedelta(hours=h1, minutes=m1)
    try:
        events, problems = _load(lo, hi, tz)
    except LookupError:
        return NOT_CONNECTED
    busy = sorted((max(x["s"], lo), min(x["e"], hi)) for x in events
                  if not x["all_day"] and (x.get("transp") or "").upper() != "TRANSPARENT" and x["e"] > x["s"])
    free, cursor = [], lo
    for b0, b1 in busy:
        if b0 > cursor:
            free.append((cursor, b0))
        cursor = max(cursor, b1)
    if cursor < hi:
        free.append((cursor, hi))
    gaps = [{"from": a.strftime("%H:%M"), "to": b.strftime("%H:%M"), "minutes": int((b - a).total_seconds() // 60)}
            for a, b in free if (b - a) >= timedelta(minutes=int(min_minutes))]
    return {
        "date": d.strftime("%a %d %b %Y"), "hours": f"{day_start}–{day_end} ({tz})",
        "free": gaps, "busy_events": len(busy),
        "all_day": [x.get("summary") or "(no title)" for x in events if x["all_day"]],
        **({"problems": problems} if problems else {}),
    }


def tools() -> list[Tool]:
    return [
        Tool("calendar_events", "The user's calendar events (read-only), recurring events expanded. Use for "
             "'what's on today/this week', 'when is my next X', 'am I busy on…'.",
             {"start": "str, first day: YYYY-MM-DD, today, tomorrow or a weekday name; optional (default today)",
              "days": f"int, how many days from start (max {MAX_DAYS}); optional (default 7)",
              "query": "str, only events whose title, location or notes contain this; optional",
              "limit": "int, max events returned; optional (default 50)"},
             calendar_events, untrusted=True),
        Tool("free_slots", "Free gaps in the user's calendar on one day, within working hours.",
             {"date": "str, YYYY-MM-DD, today, tomorrow or a weekday name; optional (default today)",
              "day_start": "str, HH:MM; optional (default 09:00)", "day_end": "str, HH:MM; optional (default 18:00)",
              "min_minutes": "int, shortest gap worth listing; optional (default 30)"},
             free_slots, untrusted=True),
    ]
