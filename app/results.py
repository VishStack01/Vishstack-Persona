"""Pull real results for YOUR OWN published posts, so calibration doesn't depend on typing numbers in.

- YouTube: views, likes and comments from the YouTube Data API (public counts, any video).
- Instagram: likes and comments for your own media (Instagram API token). Reach and views come from
  the insights endpoint when your token and account type allow it; metric names change over time,
  so missing insights are reported, never guessed.
Hold rate and conversions (downloads, purchases) aren't available from these APIs: add them by hand
from YouTube Studio, Instagram insights or your app store console.
"""
import os
import re

from .search import SearchError, YT, _client, _key, _ok

YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")


def _norm(url: str) -> str:
    return url.split("?")[0].split("#")[0].rstrip("/").lower().replace("www.", "")


def youtube(url: str) -> dict:
    m = YT_ID.search(url)
    if not m:
        raise SearchError("That doesn't look like a YouTube video link.")
    with _client() as c:
        r = c.get(f"{YT}/videos", params={"part": "statistics", "id": m.group(1), "key": _key("YOUTUBE_API_KEY")})
        _ok(r, "YouTube")
        items = r.json().get("items", [])
    if not items:
        raise SearchError("YouTube didn't find that video. Is it public?")
    st = items[0].get("statistics", {})
    return {"platform": "YouTube", "views": int(st.get("viewCount", 0) or 0),
            "likes": int(st.get("likeCount", 0) or 0), "comments": int(st.get("commentCount", 0) or 0)}


def instagram(url: str) -> dict:
    token = _key("INSTAGRAM_ACCESS_TOKEN")
    base = f"https://graph.instagram.com/{os.getenv('IG_GRAPH_VERSION', 'v23.0')}"
    want = _norm(url)
    with _client() as c:
        r = c.get(f"{base}/me/media", params={"fields": "id,permalink,like_count,comments_count", "limit": 50,
                                              "access_token": token})
        _ok(r, "Instagram")
        media = next((m for m in r.json().get("data", []) if _norm(m.get("permalink", "")) == want), None)
        if not media:
            raise SearchError("That post isn't among your 50 latest Instagram posts. Check the link, or add the numbers by hand.")
        out = {"platform": "Instagram", "likes": media.get("like_count", 0) or 0,
               "comments": media.get("comments_count", 0) or 0, "views": None, "reach": None, "note": ""}
        ri = c.get(f"{base}/{media['id']}/insights", params={"metric": "views,reach", "access_token": token})
        if ri.status_code < 400:
            for d in ri.json().get("data", []):
                vals = d.get("values") or []
                v = vals[0].get("value") if vals else d.get("total_value", {}).get("value")
                if d.get("name") in ("views", "reach") and isinstance(v, (int, float)):
                    out[d["name"]] = int(v)
        else:
            out["note"] = "Instagram didn't return views or reach for this post. Add views by hand from your insights."
    return out


def fetch(url: str) -> dict:
    host = _norm(url)
    if "youtube.com" in host or "youtu.be" in host:
        return youtube(url)
    if "instagram.com" in host:
        return instagram(url)
    raise SearchError("Paste the link to your own YouTube or Instagram post.")
