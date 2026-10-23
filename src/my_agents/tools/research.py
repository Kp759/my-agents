"""Paper search and trend signals.

search_papers: arXiv first; falls back to OpenAlex when arXiv fails or rate-limits.
trending_papers: Hugging Face daily papers (community-upvoted new arXiv papers).
No keys needed. Set OPENALEX_API_KEY for a larger OpenAlex budget.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from . import Tool
from .http import get, get_json

ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _clip(text: str | None, n: int = 400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def search_arxiv(query: str, limit: int = 8, sort: str = "relevance") -> list[dict]:
    data = get("https://export.arxiv.org/api/query", {
        "search_query": f"all:{query}",
        "max_results": limit,
        "sortBy": "submittedDate" if sort == "recent" else "relevance",
        "sortOrder": "descending",
    })
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", ATOM):
        authors = [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)]
        out.append({
            "title": _clip(e.findtext("a:title", "", ATOM), 300),
            "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
            "year": int(e.findtext("a:published", "0", ATOM)[:4] or 0),
            "url": e.findtext("a:id", "", ATOM).replace("http://", "https://"),
            "abstract": _clip(e.findtext("a:summary", "", ATOM)),
            "source": "arxiv",
        })
    return out


def _openalex_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    words = sorted((i, w) for w, idxs in inverted.items() for i in idxs)
    return " ".join(w for _, w in words)


def search_openalex(query: str, limit: int = 8, sort: str = "relevance") -> list[dict]:
    data = get_json("https://api.openalex.org/works", {
        "search": query,
        "per_page": limit,
        "sort": "publication_date:desc" if sort == "recent" else None,
        "select": "title,authorships,publication_year,doi,ids,cited_by_count,abstract_inverted_index",
        "api_key": os.environ.get("OPENALEX_API_KEY"),
    })
    out = []
    for w in data.get("results", []):
        names = [a["author"]["display_name"] for a in w.get("authorships", [])]
        ids = w.get("ids") or {}
        out.append({
            "title": _clip(w.get("title"), 300),
            "authors": ", ".join(names[:3]) + (" et al." if len(names) > 3 else ""),
            "year": w.get("publication_year"),
            "url": w.get("doi") or ids.get("openalex", ""),
            "citations": w.get("cited_by_count"),
            "abstract": _clip(_openalex_abstract(w.get("abstract_inverted_index"))),
            "source": "openalex",
        })
    return out


def search_papers(query: str, limit: int = 8, sort: str = "relevance") -> dict:
    limit = max(1, min(int(limit), 20))
    errors = []
    for name, fn in (("arxiv", search_arxiv), ("openalex", search_openalex)):
        try:
            results = fn(query, limit, sort)
            if results:
                return {"source": name, "results": results, **({"fallback_reason": errors} if errors else {})}
        except Exception as e:  # network errors and rate limits: try the next source
            errors.append(f"{name}: {e}")
    return {"results": [], "errors": errors}


def trending_papers(limit: int = 10, keyword: str | None = None, date: str | None = None) -> list[dict]:
    rows = get_json("https://huggingface.co/api/daily_papers", {"limit": 100, "date": date})
    out = []
    for r in rows:
        p = r.get("paper", {})
        text = f"{p.get('title', '')} {p.get('summary', '')}".lower()
        if keyword and keyword.lower() not in text:
            continue
        out.append({
            "title": _clip(p.get("title"), 300),
            "url": f"https://arxiv.org/abs/{p['id']}" if p.get("id") else None,
            "upvotes": p.get("upvotes", 0),
            "published": (r.get("publishedAt") or "")[:10],
            "abstract": _clip(p.get("summary"), 300),
        })
    out.sort(key=lambda x: x["upvotes"], reverse=True)
    return out[: max(1, min(int(limit), 30))]


def tools() -> list[Tool]:
    return [
        Tool("search_papers", "Search the literature (arXiv, falling back to OpenAlex). Returns title, authors, "
             "year, url, abstract. Only cite papers returned by this tool.",
             {"query": "str", "limit": "int, max 20, optional", "sort": "str, relevance | recent, optional"},
             search_papers, untrusted=True),
        Tool("trending_papers", "Papers the ML community is upvoting on Hugging Face daily papers, "
             "most upvoted first.",
             {"limit": "int, optional", "keyword": "str, filter on title/abstract, optional",
              "date": "str, YYYY-MM-DD for a specific day, optional"}, trending_papers, untrusted=True),
    ]
