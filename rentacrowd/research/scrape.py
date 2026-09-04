"""Free, keyless scrapers for real customer language.

Sources, all public and keyless:
  - Apple iTunes Search + RSS customer-reviews JSON (official Apple endpoints).
  - Hacker News via the Algolia search API (free, keyless) - stories + comments.
  - Reddit, but ONLY if REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are set in the
    environment. Reddit locked down its public `.json` in 2023 (returns 403), so
    it now needs a free OAuth app; without those vars Reddit is simply skipped.

Everything is best-effort: a source that errors or returns nothing is skipped,
and the study proceeds on whatever evidence was gathered (possibly none).
"""

from __future__ import annotations

import os
import time

import requests

_UA = "RentACrowd/0.1 (internal synthetic-research tool; contact: research@localhost)"
_TIMEOUT = 15


def _get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Apple
# --------------------------------------------------------------------------- #


def itunes_lookup(term: str) -> dict | None:
    """Resolve a product name to an App Store app (id + real name)."""
    data = _get(
        "https://itunes.apple.com/search",
        {"term": term, "entity": "software", "limit": 1, "country": "us"},
    )
    if not data or not data.get("results"):
        return None
    app = data["results"][0]
    return {"id": app["trackId"], "name": app["trackName"], "seller": app.get("sellerName", "")}


def itunes_reviews(app_id: int, pages: int = 3) -> list[dict]:
    out: list[dict] = []
    for page in range(1, pages + 1):
        data = _get(
            f"https://itunes.apple.com/us/rss/customerreviews/page={page}/id={app_id}"
            "/sortby=mostrecent/json"
        )
        entries = (data or {}).get("feed", {}).get("entry", []) or []
        for e in entries:
            if "im:rating" not in e:  # first entry is app metadata
                continue
            out.append(
                {
                    "rating": int(e["im:rating"]["label"]),
                    "title": e.get("title", {}).get("label", ""),
                    "text": e.get("content", {}).get("label", ""),
                }
            )
        time.sleep(0.3)
    return out


# --------------------------------------------------------------------------- #
# Hacker News (Algolia)
# --------------------------------------------------------------------------- #


def hn_mentions(query: str, limit: int = 20) -> list[dict]:
    data = _get(
        "https://hn.algolia.com/api/v1/search",
        {"query": query, "tags": "(story,comment)", "hitsPerPage": limit},
    )
    out = []
    for h in (data or {}).get("hits", []):
        text = (h.get("comment_text") or h.get("story_text") or h.get("title") or "").strip()
        if not text:
            continue
        out.append({"source": "hackernews", "text": text[:1400], "points": h.get("points") or 0})
    return out


# --------------------------------------------------------------------------- #
# Reddit (opt-in: needs a free OAuth app)
# --------------------------------------------------------------------------- #


def _reddit_token() -> str | None:
    cid, secret = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        return None
    try:
        r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(cid, secret),
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
        )
        return r.json().get("access_token") if r.status_code == 200 else None
    except Exception:
        return None


def reddit_mentions(query: str, limit: int = 25) -> list[dict]:
    token = _reddit_token()
    if not token:
        return []
    try:
        r = requests.get(
            "https://oauth.reddit.com/search",
            params={"q": query, "limit": limit, "sort": "relevance", "t": "year"},
            headers={"Authorization": f"Bearer {token}", "User-Agent": _UA},
            timeout=_TIMEOUT,
        )
        posts = r.json().get("data", {}).get("children", []) if r.status_code == 200 else []
    except Exception:
        posts = []
    out = []
    for p in posts:
        d = p.get("data", {})
        out.append(
            {
                "source": f"r/{d.get('subreddit', '')}",
                "text": (d.get("title", "") + "\n" + (d.get("selftext") or "")).strip()[:1400],
                "points": d.get("score", 0),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def gather(competitor: str) -> dict:
    """Everything we could find about one competitor. Cheap, no LLM."""
    app = itunes_lookup(competitor)
    reviews = itunes_reviews(app["id"]) if app else []
    discussion = hn_mentions(competitor) + reddit_mentions(competitor)
    return {
        "competitor": competitor,
        "app_match": app,
        "reviews": reviews,
        "discussion": discussion,
        "counts": {"itunes_reviews": len(reviews), "discussion_posts": len(discussion)},
    }
