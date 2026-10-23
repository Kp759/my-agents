"""Live web tools, offline: parsers run on saved markup, the network is faked."""

import base64
from datetime import datetime

from my_agents.adapters import FakeAdapter
from my_agents.agents import AgentSpec, RouterConfig, load_agents, load_router_config
from my_agents.graph import build_graph
from my_agents.store import Store
from my_agents.tools import web
from my_agents.tools.http import HTTPStatusError
from my_agents.tracing import NullTracer

DDG = """
<div class="result results_links results_links_deep result--ad ">
  <a rel="nofollow" class="result__a" href="https://ads.example/">Sponsored</a></div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title"><a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Flangchain-ai%2Flanggraph%2Freleases&amp;rut=x">
     Releases &middot; <b>langgraph</b></a></h2>
  <a class="result__snippet" href="#">Latest: <b>1.2.16</b> released 3 days ago</a>
</div>
"""

BING_TARGET = "https://docs.langchain.com/oss/python/langgraph/overview"
BING = f"""
<ol id="b_results"><li class="b_algo" data-id><div class="b_tpcn">logo</div>
<h2><a target="_blank" href="https://www.bing.com/ck/a?!&amp;&amp;p=abc&amp;u=a1{
    base64.urlsafe_b64encode(BING_TARGET.encode()).decode().rstrip("=")}&amp;ntb=1"><strong>LangGraph</strong> overview</a></h2>
<div class="b_caption"><p class="b_lineclamp2">Build <strong>stateful</strong> agents.</p></div></li></ol>
"""


def test_duckduckgo_results_are_parsed_and_ads_skipped(monkeypatch):
    monkeypatch.setattr(web, "get", lambda url, params=None: DDG.encode())
    out = web.web_search("langgraph release", deep=False)
    assert out["engine"] == "duckduckgo"
    assert out["results"] == [{"title": "Releases · langgraph", "url": "https://github.com/langchain-ai/langgraph/releases",
                               "snippet": "Latest: 1.2.16 released 3 days ago"}]


def test_bing_is_the_fallback_when_duckduckgo_refuses(monkeypatch):
    def fake_get(url, params=None):
        if "duckduckgo" in url:
            raise HTTPStatusError(202, url)  # its rate-limit / bot-check answer
        assert params["filters"] == 'ex1:"ez2"'  # recency=week
        return BING.encode()

    monkeypatch.setattr(web, "get", fake_get)
    out = web.web_search("langgraph", recency="week", deep=False)
    assert out["engine"] == "bing"
    assert out["results"][0] == {"title": "LangGraph overview", "url": BING_TARGET, "snippet": "Build stateful agents."}


def test_search_failure_explains_alternatives(monkeypatch):
    monkeypatch.setattr(web, "get", lambda url, params=None: b"<html>please verify</html>")
    try:
        web.web_search("x")
    except RuntimeError as e:
        assert "tech_news" in str(e) and "never try to bypass" in str(e)
    else:
        raise AssertionError("expected a failure")


def deep_search(monkeypatch, pages):
    """web_search with DuckDuckGo results for `pages` (url -> (content-type, body or exception))."""
    links = "".join(f'<div class="result "><a class="result__a" href="{u}">T{i}</a></div>' for i, u in enumerate(pages))
    monkeypatch.setattr(web, "get", lambda url, params=None: links.encode())

    def download(url, timeout=20):
        ctype, body = pages[url]
        if isinstance(body, Exception):
            raise body
        return url, {"content-type": ctype}, body

    monkeypatch.setattr(web, "_download", download)
    return web.web_search("langgraph durable execution release")


def test_deep_search_returns_matching_passages_from_the_top_pages(monkeypatch):
    page = (b"<html><head><title>Notes</title><meta name=date content=2026-10-16></head><body>"
            b"<p>Cookie banner text that is long enough to count as a paragraph here.</p>"
            b"<p>LangGraph 1.2 adds durable execution, so runs survive restarts.</p></body></html>")
    out = deep_search(monkeypatch, {"https://a.example/notes": ("text/html", page)})
    [p] = out["pages"]
    assert p["url"] == "https://a.example/notes" and p["published"] == {"date": "2026-10-16"}
    assert p["passages"] == "LangGraph 1.2 adds durable execution, so runs survive restarts."  # banner dropped
    assert "Answer from `pages`" in out["next"]


def test_deep_search_skips_failures_and_bot_checks_and_stops_at_three(monkeypatch):
    ok = ("text/plain", b"LangGraph release notes: durable execution is now stable in this release.")
    out = deep_search(monkeypatch, {
        "https://blocked.example/": ("text/html", b"<title>Client Challenge</title><p>Enable JavaScript</p>"),
        "https://down.example/": ("text/html", HTTPStatusError(503, "https://down.example/")),
        "https://a.example/": ok, "https://b.example/": ok, "https://c.example/": ok,
    })
    assert [p["url"] for p in out["pages"]] == ["https://a.example/", "https://b.example/"]  # 4 tried: 3 pages + 1 spare
    reasons = {s["url"]: s["reason"] for s in out["skipped"]}
    assert "bot check" in reasons["https://blocked.example/"] and "503" in reasons["https://down.example/"]


def test_snippets_only_when_deep_is_off(monkeypatch):
    def no_download(*a, **k):
        raise AssertionError("deep=False must not open pages")
    monkeypatch.setattr(web, "_download", no_download)
    monkeypatch.setattr(web, "get", lambda url, params=None: DDG.encode())
    assert "pages" not in web.web_search("x", deep=False)


def test_passages_fall_back_to_the_page_opening():
    assert web._passages("Nothing relevant here at all in this rather long paragraph.", "quantum") .startswith("Nothing")


def test_pages_become_readable_text_with_dates(monkeypatch):
    html = b"""<html><head><title>Release notes</title>
      <meta property="article:published_time" content="2026-10-16T00:00:00Z"><script>var x=1</script></head>
      <body><nav>Home | Docs</nav><h1>v1.2</h1><p>Adds   durable
      execution.</p><footer>(c) site</footer></body></html>"""
    monkeypatch.setattr(web, "_download", lambda url, timeout=20: (url, {"content-type": "text/html; charset=utf-8"}, html))
    page = web.web_fetch("https://example.org/notes")
    assert page["title"] == "Release notes"
    assert page["published"] == {"article:published_time": "2026-10-16T00:00:00Z"}
    assert page["text"] == "v1.2\n\nAdds durable execution."  # no script, nav or footer
    assert page["next_offset"] is None and "note" not in page


def test_long_pages_come_in_parts(monkeypatch):
    body = ("word " * 2000).encode()  # 10,000 chars: two parts
    monkeypatch.setattr(web, "_download", lambda url, timeout=20: (url, {"content-type": "text/plain"}, body))
    first = web.web_fetch("https://example.org/long")
    second = web.web_fetch("https://example.org/long", offset=first["next_offset"])
    assert len(first["text"]) == web.PAGE_CHARS and second["next_offset"] is None


def test_binary_files_are_refused(monkeypatch):
    monkeypatch.setattr(web, "_download", lambda url, timeout=20: (url, {"content-type": "application/pdf"}, b"%PDF"))
    try:
        web.web_fetch("https://example.org/paper.pdf")
    except ValueError as e:
        assert "not a text page" in str(e)
    else:
        raise AssertionError("expected a refusal")


def test_package_info_reports_latest_version(monkeypatch):
    data = {"info": {"version": "1.2.16", "summary": "graphs", "project_urls": {}, "requires_python": ">=3.10"},
            "releases": {"1.2.15": [{"upload_time_iso_8601": "2026-10-01T00:00:00Z"}],
                         "1.2.16": [{"upload_time_iso_8601": "2026-10-16T00:00:00Z"}], "0.0.1": []}}
    monkeypatch.setattr(web, "get_json", lambda url, params=None: data)
    info = web.package_info("langgraph")
    assert info["latest"] == "1.2.16" and info["released"] == "2026-10-16"
    assert [r["version"] for r in info["recent_releases"]] == ["1.2.16", "1.2.15"]


def test_every_prompt_states_todays_date():
    model = FakeAdapter()
    router = RouterConfig("fake", None, "a", freshness="## Today is {today}\nCheck before you trust your memory.")
    graph = build_graph({"a": AgentSpec("a", "x", "You help.", "fake")}, router, NullTracer(),
                        store=Store(":memory:"), adapters={"a": model}, router_adapter=FakeAdapter())
    graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, {"configurable": {"thread_id": "t"}})
    system = model.calls[0][0]
    now = datetime.now()
    assert f"## Today is {now:%A} {now.day} {now:%B %Y}" in system
    assert system.index("Today is") < system.index("You help.")


def test_shipped_config_gives_knowledge_agents_live_web_access():
    assert "{today}" in load_router_config().freshness
    agents = load_agents()
    for name in ("orchestrator", "explainer", "lit_surveyor", "trend_scout", "career_coach", "coder", "writer"):
        assert {"web_search", "web_fetch"} <= set(agents[name].tools), name
