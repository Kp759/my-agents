"""The iCalendar reader behind @friday: parsing, recurrence, time zones and the two tools."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from my_agents.tools import calendar

NY = ZoneInfo("America/New_York")

ICS = """BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VTIMEZONE\r
TZID:America/New_York\r
END:VTIMEZONE\r
BEGIN:VEVENT\r
UID:standup\r
DTSTART;TZID=America/New_York:20261019T090000\r
DTEND;TZID=America/New_York:20261019T091500\r
RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR\r
EXDATE;TZID=America/New_York:20261021T090000\r
SUMMARY:Lab stand-up\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:standup\r
RECURRENCE-ID;TZID=America/New_York:20261023T090000\r
DTSTART;TZID=America/New_York:20261023T110000\r
DTEND;TZID=America/New_York:20261023T111500\r
SUMMARY:Lab stand-up (moved)\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:advisor\r
DTSTART:20261022T190000Z\r
DURATION:PT1H\r
SUMMARY:Advisor meeting\\, weekly sync\r
LOCATION:GITC 4400\r
DESCRIPTION:Bring the stop-signal pilot results.\\nAnd the plan for E\r
 1.\r
BEGIN:VALARM\r
TRIGGER:-PT10M\r
DESCRIPTION:This is the alarm, not the event\r
END:VALARM\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:conf\r
DTSTART;VALUE=DATE:20261026\r
DTEND;VALUE=DATE:20261029\r
SUMMARY:Workshop trip\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:seminar\r
DTSTART;TZID=America/New_York:20260113T160000\r
DTEND;TZID=America/New_York:20260113T170000\r
RRULE:FREQ=MONTHLY;BYDAY=2TU\r
SUMMARY:MAS reading group\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:report\r
DTSTART;VALUE=DATE:20260130\r
RRULE:FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=-1\r
SUMMARY:Monthly report due\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:course\r
DTSTART;TZID=America/New_York:20261001T100000\r
DTEND;TZID=America/New_York:20261001T113000\r
RRULE:FREQ=WEEKLY;COUNT=4\r
SUMMARY:Short course\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:gone\r
DTSTART;TZID=America/New_York:20261022T130000\r
DTEND;TZID=America/New_York:20261022T140000\r
STATUS:CANCELLED\r
SUMMARY:Cancelled talk\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:focus\r
DTSTART;TZID=America/New_York:20261022T150000\r
DTEND;TZID=America/New_York:20261022T160000\r
TRANSP:TRANSPARENT\r
SUMMARY:Focus time (shows as free)\r
END:VEVENT\r
END:VCALENDAR\r
"""


def window(start: date, days: int):
    s = datetime.combine(start, datetime.min.time(), tzinfo=NY)
    return calendar.expand(calendar.parse_ics(ICS, "njit", NY), s, s + timedelta(days=days), NY)


def titles(events):
    return [(e["s"].strftime("%a %d %H:%M"), e["summary"]) for e in events]


def test_parses_text_escapes_folding_and_ignores_alarms():
    ev = next(e for e in calendar.parse_ics(ICS, "njit", NY) if e["uid"] == "advisor")
    assert ev["summary"] == "Advisor meeting, weekly sync"
    assert ev["description"] == "Bring the stop-signal pilot results.\nAnd the plan for E1."
    assert ev["start"] == datetime(2026, 10, 22, 19, tzinfo=calendar.timezone.utc)


def test_week_view_expands_recurrence_exdates_moves_and_cancellations():
    got = titles(window(date(2026, 10, 19), 7))
    assert ("Mon 19 09:00", "Lab stand-up") in got
    assert not any(t.startswith("Wed 21") and s == "Lab stand-up" for t, s in got)  # EXDATE
    assert ("Fri 23 11:00", "Lab stand-up (moved)") in got and ("Fri 23 09:00", "Lab stand-up") not in got
    assert ("Thu 22 15:00", "Advisor meeting, weekly sync") in got  # 19:00 UTC is 15:00 in New York
    assert "Cancelled talk" not in [s for _, s in got]
    starts = [e["s"] for e in window(date(2026, 10, 19), 7)]
    assert starts == sorted(starts)


def test_all_day_multi_day_event_overlaps_each_day_but_not_the_next():
    assert "Workshop trip" in [e["summary"] for e in window(date(2026, 10, 28), 1)]
    assert "Workshop trip" not in [e["summary"] for e in window(date(2026, 10, 29), 1)]


def test_monthly_ordinal_weekday_and_setpos():
    oct_ = window(date(2026, 10, 1), 31)
    reading = [e["s"].day for e in oct_ if e["summary"] == "MAS reading group"]
    report = [e["s"].day for e in oct_ if e["summary"] == "Monthly report due"]
    assert reading == [13]  # 2nd Tuesday of October 2026
    assert report == [30]  # last weekday of October 2026 (a Friday)
    nov = window(date(2026, 11, 1), 30)
    assert [e["s"].day for e in nov if e["summary"] == "Monthly report due"] == [30]  # Monday 30 Nov


def test_count_limits_occurrences():
    days = [e["s"].day for e in window(date(2026, 10, 1), 40) if e["summary"] == "Short course"]
    assert days == [1, 8, 15, 22]


def test_recurring_local_time_survives_daylight_saving_change():
    # US clocks go back on 1 Nov 2026: the stand-up stays at 09:00 local time.
    nov = [e for e in window(date(2026, 10, 30), 7) if e["summary"] == "Lab stand-up"]
    assert {e["s"].strftime("%H:%M") for e in nov} == {"09:00"}
    assert {e["s"].utcoffset() for e in nov} == {timedelta(hours=-4), timedelta(hours=-5)}


def test_until_bounds_the_rule():
    ics = ICS.replace("RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR", "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR;UNTIL=20261021T235959Z")
    s = datetime(2026, 10, 19, tzinfo=NY)
    evs = calendar.expand(calendar.parse_ics(ics, "c", NY), s, s + timedelta(days=14), NY)
    assert [e["s"].day for e in evs if e["summary"] == "Lab stand-up"] == [19]  # 21st is an EXDATE


@pytest.fixture
def connected(tmp_path, monkeypatch):
    feed = tmp_path / "njit.ics"
    feed.write_text(ICS)
    monkeypatch.setenv("MY_AGENTS_CALENDARS", str(tmp_path / "calendars.txt"))
    monkeypatch.setenv("TZ", "America/New_York")
    calendar._cache.clear()
    calendar.save_source("njit", str(feed))
    return feed


def test_calendar_events_tool(connected):
    out = calendar.calendar_events("2026-10-22", days=1)
    assert [e["title"] for e in out["events"]] == ["Short course", "Advisor meeting, weekly sync",
                                                   "Focus time (shows as free)"]
    first = out["events"][1]
    assert first["when"] == "Thu 22 Oct 15:00–16:00" and first["location"] == "GITC 4400"
    assert out["events"][0]["recurring"] and out["events"][2]["shows_as"] == "free"
    assert calendar.calendar_events("2026-10-19", days=14, query="stand-up")["total"] == 5  # 19, 23 (moved), 26, 28, 30


def test_free_slots_ignores_free_and_all_day_events(connected):
    out = calendar.free_slots("2026-10-22")
    assert out["free"] == [{"from": "09:00", "to": "10:00", "minutes": 60},  # course 10:00-11:30
                           {"from": "11:30", "to": "15:00", "minutes": 210},  # advisor 15-16; focus time is "free"
                           {"from": "16:00", "to": "18:00", "minutes": 120}]
    out = calendar.free_slots("2026-10-26", day_start="08:00", day_end="10:00")
    assert out["all_day"] == ["Workshop trip"]
    assert out["free"] == [{"from": "08:00", "to": "09:00", "minutes": 60},
                           {"from": "09:15", "to": "10:00", "minutes": 45}]


def test_not_connected_and_broken_feed(tmp_path, monkeypatch, connected):
    monkeypatch.setenv("MY_AGENTS_CALENDARS", str(tmp_path / "none.txt"))
    assert "my-agents calendar add" in calendar.calendar_events()["note"]
    monkeypatch.setenv("MY_AGENTS_CALENDARS", str(tmp_path / "calendars.txt"))
    bad = tmp_path / "bad.ics"
    bad.write_text("not a calendar")
    calendar.save_source("bad", str(bad))
    out = calendar.calendar_events("2026-10-22", days=1)
    assert out["events"] and out["problems"][0]["calendar"] == "bad"


def test_dates_and_bad_input():
    tz = NY
    assert calendar._day("2026-10-23", tz) == date(2026, 10, 23)
    assert calendar._day("tomorrow", tz) - calendar._day("today", tz) == timedelta(days=1)
    assert calendar._day("friday", tz).weekday() == 4
    with pytest.raises(ValueError):
        calendar._day("next blue moon", tz)
    with pytest.raises(ValueError):
        calendar._clock("25:99")
