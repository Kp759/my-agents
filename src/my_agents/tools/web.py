"""Live web access, so agents answer from today's sources and not from training data.

Every model has a training cutoff, so on its own it cannot know about recent releases,
papers, people or prices. These tools work the same with every backend (Claude, Codex,
Qwen), because they run here and not inside a model provider.

web_search: DuckDuckGo first, Bing when DuckDuckGo refuses (it rate-limits). No keys.
            By default it also opens the top results and returns the passages that match
            the query ("deep" search), so even a small model that never thinks to call
            web_fetch answers from the pages, not from 30-word snippets.
tech_news:  Hacker News stories with dates and points, a fast signal of what is new.
package_info: latest version and release dates of a PyPI or npm package.
web_fetch:  read one public web page as text, with its publication date when it has one.

Everything returned is untrusted (any page can contain prompt injection). web_fetch can
reach any public host, so it is the one tool that could carry data out in a URL; the
graph gates it (see `Tool.egress` and docs/SECURITY.md).
"""

from __future__ import annotations

import base64
import http.client
import re
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, wait
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

from ..guardrails import check_public_url
from . import Tool
from .http import USER_AGENT, HTTPStatusError, get, get_json

MAX_BYTES = 3_000_000  # read at most this much of a page
PAGE_CHARS = 6_000  # text returned per web_fetch call; use `offset` to read on
DEEP_PAGES = 3  # pages a deep search reads
DEEP_CHARS = 1_000  # passage characters kept per page (tool results are capped at 8,000 in total)
DEEP_TIMEOUT = 15  # seconds for all deep reads together
STOPWORDS = set("the a an and or of to in on for with what which who when where how is are was were did does "
                "do latest recent most new newest current about from by at as it its this that be".split())
MAX_REDIRECTS = 5
TEXT_TYPES = ("text/html", "text/plain", "text/markdown", "application/xhtml+xml", "application/json",
              "application/xml", "text/xml", "application/rss+xml", "application/atom+xml")


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", html))).strip()


# ---- search ---------------------------------------------------------------------------


def _ddg(query: str, recency: str | None) -> list[dict]:
    # Plain `get` (not a browser user agent): when DuckDuckGo shows a bot check we stop, never solve it.
    page = get("https://html.duckduckgo.com/html/",
               {"q": query, "df": {"day": "d", "week": "w", "month": "m", "year": "y"}.get(recency or "")})
    out = []
    for block in page.decode("utf-8", "replace").split('<div class="result ')[1:]:
        if "result--ad" in block.split(">", 1)[0]:
            continue
        link = re.search(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not link:
            continue
        href = unescape(link.group(1))
        target = parse_qs(urlparse(href).query).get("uddg", [href])[0]
        snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        out.append({"title": _text(link.group(2)), "url": urljoin("https:", target),
                    "snippet": _text(snippet.group(1)) if snippet else ""})
    return out


def _bing_target(href: str) -> str:
    # Bing wraps results in a click-tracking link whose `u` parameter is "a1" + base64url(target).
    u = parse_qs(urlparse(href).query).get("u", [""])[0]
    if u.startswith("a1"):
        try:
            return base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
        except ValueError:
            pass
    return href


def _bing(query: str, recency: str | None) -> list[dict]:
    window = {"day": "ez1", "week": "ez2", "month": "ez3"}.get(recency or "")
    page = get("https://www.bing.com/search",
               {"q": query, "setlang": "en", "filters": f'ex1:"{window}"' if window else None})
    out = []
    for block in page.decode("utf-8", "replace").split('<li class="b_algo"')[1:]:
        link = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not link:
            continue
        snippet = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        out.append({"title": _text(link.group(2)), "url": _bing_target(unescape(link.group(1))),
                    "snippet": _text(snippet.group(1)) if snippet else ""})
    return out


def _passages(text: str, query: str, budget: int = DEEP_CHARS) -> str:
    """The paragraphs that mention the most query words, kept in page order."""
    terms = {w for w in re.findall(r"[a-z0-9][a-z0-9.+-]*", query.lower()) if w not in STOPWORDS and len(w) > 1}
    paras = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) >= 40]
    scored = sorted(((sum(t in p.lower() for t in terms), -i, p) for i, p in enumerate(paras)), reverse=True)
    picked, used = [], 0
    for score, neg_i, p in scored:
        if score == 0 or used >= budget:
            break
        p = p[: budget - used]
        picked.append((-neg_i, p))
        used += len(p)
    if not picked:  # nothing matched: the page's opening is the best guess
        return text[:budget // 2].strip()
    return "\n…\n".join(p for _, p in sorted(picked))


def _deep_read(results: list[dict], query: str) -> tuple[list[dict], list[dict]]:
    """Open the top results in parallel; return (pages, skipped) in result order."""
    candidates = results[:DEEP_PAGES + 1]  # one spare in case a page fails
    pool = ThreadPoolExecutor(max_workers=len(candidates))
    futures = [pool.submit(_read, r["url"], 8) for r in candidates]
    wait(futures, timeout=DEEP_TIMEOUT)
    pool.shutdown(wait=False, cancel_futures=True)
    pages, skipped = [], []
    for r, f in zip(candidates, futures):
        if len(pages) == DEEP_PAGES:
            break
        if not f.done():
            skipped.append({"url": r["url"], "reason": "timed out"})
            continue
        try:
            final, headers, title, dates, text, blocked = f.result()
        except Exception as e:  # blocked address, HTTP error, binary file...
            skipped.append({"url": r["url"], "reason": f"{type(e).__name__}: {e}"[:200]})
            continue
        if blocked:
            skipped.append({"url": r["url"], "reason": "bot check or JavaScript-only page (not bypassed)"})
            continue
        pages.append({"url": final, "title": title or r["title"], "published": dates or None,
                      "last_modified": headers.get("last-modified"), "passages": _passages(text, query)})
    return pages, skipped


def web_search(query: str, limit: int = 8, recency: str | None = None, deep: bool = True) -> dict:
    if recency not in (None, "", "day", "week", "month", "year"):
        raise ValueError("recency must be day, week, month or year")
    limit, errors = max(5, min(int(limit), 15)), []  # at least 5: one result is too easy to over-trust
    for engine, search in (("duckduckgo", _ddg), ("bing", _bing)):
        try:
            results = search(query, recency)
        except (HTTPStatusError, OSError) as e:
            errors.append(f"{engine}: {e}")
            continue
        if results:
            out = {"engine": engine, "searched_at": time.strftime("%Y-%m-%d %H:%M %Z"), "results": results[:limit]}
            if deep and str(deep).lower() != "false":
                out["pages"], out["skipped"] = _deep_read(results, query)
                out["next"] = ("Answer from `pages` (text read from the top results just now), citing their URLs "
                               "and dates. web_fetch a page if you need more of it.")
            else:
                out["next"] = "Snippets can be stale or cut off: web_fetch the best 1-3 results before answering."
            return out
        errors.append(f"{engine}: no results (or a bot check, which we never try to bypass)")
    raise RuntimeError("web search failed: " + "; ".join(errors) +
                       ". Try tech_news, search_papers or github_search_repos, or web_fetch a known URL.")


def tech_news(query: str, days: int = 30, limit: int = 10) -> list[dict]:
    """Hacker News stories matching `query` from the last `days` days, most relevant first."""
    since = int(time.time()) - max(1, min(int(days), 3650)) * 86400
    data = get_json("https://hn.algolia.com/api/v1/search", {
        "query": query, "tags": "story", "hitsPerPage": max(1, min(int(limit), 30)),
        "numericFilters": f"created_at_i>{since}",
    })
    return [{"title": h.get("title"), "url": h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
             "date": (h.get("created_at") or "")[:10], "points": h.get("points"),
             "comments": h.get("num_comments"), "discussion": f"https://news.ycombinator.com/item?id={h['objectID']}"}
            for h in data.get("hits", [])]


def package_info(name: str, ecosystem: str = "pypi") -> dict:
    """Latest version, release dates and links, straight from the package registry."""
    if not re.fullmatch(r"(@[a-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]{0,100}", name):
        raise ValueError(f"not a package name: {name!r}")
    if ecosystem == "pypi":
        data = get_json(f"https://pypi.org/pypi/{name}/json")
        info = data["info"]
        dates = {v: min(f["upload_time_iso_8601"] for f in files)[:10]
                 for v, files in data.get("releases", {}).items() if files}
        latest, summary, links = info["version"], info.get("summary"), info.get("project_urls") or {}
        requires = info.get("requires_python")
    elif ecosystem == "npm":
        data = get_json(f"https://registry.npmjs.org/{name.replace('/', '%2F')}")
        dates = {v: t[:10] for v, t in (data.get("time") or {}).items() if v not in ("created", "modified")}
        latest, summary = data["dist-tags"]["latest"], data.get("description")
        links, requires = {"homepage": data.get("homepage"), "repository": (data.get("repository") or {}).get("url")}, None
    else:
        raise ValueError("ecosystem must be pypi or npm")
    recent = sorted(dates.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {"package": name, "ecosystem": ecosystem, "latest": latest, "released": dates.get(latest),
            "summary": summary, "requires_python": requires, "links": links,
            "recent_releases": [{"version": v, "date": d} for v, d in recent],
            "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")}


# ---- fetch ----------------------------------------------------------------------------


class _PinnedHTTPS(http.client.HTTPSConnection):
    """Connect to the IP address we checked, not whatever DNS says a moment later (DNS rebinding)."""

    def __init__(self, host: str, ip: str, timeout: int):
        super().__init__(host, 443, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)  # certificate checked for host


def _download(url: str, timeout: int = 20) -> tuple[str, dict, bytes]:
    for _ in range(MAX_REDIRECTS + 1):
        host, ip = check_public_url(url)  # every hop: https, public address only
        u = urlparse(url)
        conn = _PinnedHTTPS(host, ip, timeout)
        try:
            conn.request("GET", (u.path or "/") + (f"?{u.query}" if u.query else ""), headers={
                "User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
                "Accept-Encoding": "identity", "Accept-Language": "en"})
            resp = conn.getresponse()
            headers = {k.lower(): v for k, v in resp.getheaders()}
            if resp.status in (301, 302, 303, 307, 308) and headers.get("location"):
                url = urljoin(url, headers["location"])
                continue
            if resp.status >= 400:
                raise HTTPStatusError(resp.status, url.split("?")[0])
            return url, headers, resp.read(MAX_BYTES)
        finally:
            conn.close()
    raise RuntimeError(f"more than {MAX_REDIRECTS} redirects")


class _Readable(HTMLParser):
    """Page text without scripts, styles and navigation, plus title and date metadata."""

    SKIP = {"script", "style", "noscript", "svg", "template", "nav", "footer", "iframe", "form", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
             "pre", "blockquote", "table", "ul", "ol", "dd", "dt", "hr"}
    DATE_META = {"article:published_time", "article:modified_time", "og:updated_time", "date", "dc.date",
                 "dcterms.date", "citation_publication_date", "citation_date", "pubdate", "publishdate"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self.dates, self._skip, self._pre, self._in_title = [], "", {}, 0, 0, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key in self.DATE_META and a.get("content"):
                self.dates.setdefault(key, a["content"][:40])
        elif tag == "time" and a.get("datetime"):
            self.dates.setdefault("time", a["datetime"][:40])
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        if tag == "pre":
            self._pre += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "pre" and self._pre:
            self._pre -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:  # source line breaks are just spaces, except in code blocks
            self.parts.append(data if self._pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(self.parts).split("\n"))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _read(url: str, timeout: int = 20) -> tuple[str, dict, str, dict, str, bool]:
    """Download a page; return (final url, headers, title, dates, text, looks_like_a_bot_check)."""
    final, headers, body = _download(url.strip(), timeout=timeout)
    ctype = headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype and ctype not in TEXT_TYPES:
        raise ValueError(f"not a text page ({ctype}); for papers use search_papers, for PDFs ask the user")
    charset = re.search(r"charset=([\w-]+)", headers.get("content-type", ""))
    raw = body.decode(charset.group(1) if charset else "utf-8", "replace")
    title, dates = "", {}
    if "html" in ctype or (not ctype and "<html" in raw[:2000].lower()):
        page = _Readable()
        page.feed(raw)
        text, title, dates = page.text(), re.sub(r"\s+", " ", page.title).strip(), page.dates
    else:
        text = raw
    blocked = bool(len(text) < 800 and re.search(
        r"(?i)javascript|challenge|captcha|verify you are human|couldn.t load", f"{title} {text}"))
    return final, headers, title, dates, text, blocked


def web_fetch(url: str, offset: int = 0) -> dict:
    final, headers, title, dates, text, blocked = _read(url)
    offset = max(0, int(offset))
    chunk = text[offset:offset + PAGE_CHARS]
    more = offset + PAGE_CHARS < len(text)
    return {
        "url": final, "title": title,
        "published": dates or None,  # from the page's own metadata, when it has any
        "last_modified": headers.get("last-modified"),
        "fetched_at": time.strftime("%Y-%m-%d %H:%M %Z"),
        "text": chunk,
        "chars": len(text),
        "next_offset": offset + PAGE_CHARS if more else None,  # pass as `offset` to read on
        **({"note": "This looks like a bot check or a JavaScript-only page; we never bypass those. "
                    "Use another source (search result, registry, GitHub, arXiv)."} if blocked else {}),
    }


def tools() -> list[Tool]:
    return [
        Tool("web_search", "Search the web for current information (releases, docs, news, people, events, "
             "deadlines). Returns results with snippets, and also reads the top pages and returns the passages "
             "that match your query, with their dates.",
             {"query": "str", "limit": "int, max 15, optional",
              "recency": "str, day | week | month | year, optional; restrict to recent pages",
              "deep": "bool, optional, default true; false = snippets only (faster)"},
             web_search, untrusted=True),
        Tool("tech_news", "Recent tech news from Hacker News, with dates and points: what people are "
             "discussing about a model, library, company or paper.",
             {"query": "str", "days": "int, how far back, default 30, optional", "limit": "int, optional"},
             tech_news, untrusted=True),
        Tool("web_fetch", "Read a public web page (https) as text, with its publication date when the page "
             "states one. Long pages come in parts: pass next_offset as offset to continue.",
             {"url": "str, https URL", "offset": "int, optional"}, web_fetch, untrusted=True, egress=True),
        Tool("package_info", "Latest version and recent release dates of a Python (pypi) or JavaScript (npm) "
             "package, from the registry itself. Use it before stating any version number.",
             {"name": "str", "ecosystem": "str, pypi | npm, optional"}, package_info, untrusted=True),
    ]
