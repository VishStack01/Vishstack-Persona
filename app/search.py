"""Finding existing content through official APIs and polite page fetching.

Sources and their limits:
- Web search: Brave Search API or Tavily (your key). Snippets and links.
- YouTube Data API v3: public videos with public view, like and comment counts, and public comments.
- Meta Ad Library API: ads about social issues, elections or politics, and ads delivered in the EU.
  Most commercial ads are visible on facebook.com/ads/library but are NOT returned by the API.
- Page fetch: one page at a time, only if robots.txt allows it, text extracted with trafilatura.
- Your own accounts: your Instagram media and comments (Instagram API with Instagram Login token),
  your YouTube channel's videos and comments.
No logins are bypassed and nothing is scraped from behind an account.
"""
import os
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from . import gateway

UA = os.getenv("VSP_USER_AGENT", "VishstackPersonaBot/0.1 (research; contact via site owner)")


class SearchError(Exception):
    pass


def _client(timeout: float = 30) -> httpx.Client:
    return httpx.Client(timeout=timeout, transport=gateway.TRANSPORT, headers={"User-Agent": UA}, follow_redirects=True)


def _key(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SearchError(f"Set {name} in .env to use this source.")
    return v


def available() -> dict:
    def has(*names):
        return all(os.getenv(n) for n in names)
    return {
        "web": has("BRAVE_API_KEY") or has("TAVILY_API_KEY"),
        "youtube": has("YOUTUBE_API_KEY"),
        "meta_ads": has("META_ACCESS_TOKEN"),
        "instagram": has("INSTAGRAM_ACCESS_TOKEN"),
        "fetch": True,
        "transcribe": _has_whisper(),
        "video": _has_whisper() and _has_scenes(),
    }


def _has_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _has_scenes() -> bool:
    try:
        import scenedetect  # noqa: F401
        return True
    except ImportError:
        return False


# ---------- Web ----------
def web_search(query: str, n: int = 10) -> list[dict]:
    with _client() as c:
        if os.getenv("BRAVE_API_KEY"):
            r = c.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": min(n, 20)},
                      headers={"X-Subscription-Token": os.environ["BRAVE_API_KEY"], "Accept": "application/json"})
            _ok(r, "Brave Search")
            return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
                    for x in r.json().get("web", {}).get("results", [])][:n]
        if os.getenv("TAVILY_API_KEY"):
            r = c.post("https://api.tavily.com/search", json={"query": query, "max_results": min(n, 20)},
                       headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"})
            _ok(r, "Tavily")
            return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
                    for x in r.json().get("results", [])][:n]
    raise SearchError("Set BRAVE_API_KEY or TAVILY_API_KEY in .env to search the web.")


def fetch_page(url: str) -> dict:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise SearchError("Use a full http or https link.")
    with _client() as c:
        rp = RobotFileParser()
        try:
            rr = c.get(f"{p.scheme}://{p.netloc}/robots.txt")
            rp.parse(rr.text.splitlines() if rr.status_code == 200 else [])
        except httpx.HTTPError:
            rp.parse([])
        if not rp.can_fetch(UA, url):
            raise SearchError("This site's robots.txt doesn't allow fetching that page.")
        r = c.get(url)
        _ok(r, p.netloc)
        html = r.text
    import trafilatura
    text = trafilatura.extract(html, include_comments=False) or ""
    meta = trafilatura.extract_metadata(html)
    return {"url": url, "title": (meta.title if meta and meta.title else p.netloc), "text": text[:20000]}


# ---------- YouTube ----------
YT = "https://www.googleapis.com/youtube/v3"


def youtube_search(query: str, n: int = 15, region: str = "IN", language: str | None = None,
                   published_after: str | None = None, video_duration: str | None = None) -> list[dict]:
    key = _key("YOUTUBE_API_KEY")
    params = {"part": "snippet", "q": query, "type": "video", "maxResults": min(n, 50), "regionCode": region, "key": key}
    if language:
        params["relevanceLanguage"] = language
    if published_after:
        params["publishedAfter"] = published_after
    if video_duration:  # "short" = under 4 minutes
        params["videoDuration"] = video_duration
    with _client() as c:
        r = c.get(f"{YT}/search", params=params)
        _ok(r, "YouTube")
        ids = [it["id"]["videoId"] for it in r.json().get("items", []) if it.get("id", {}).get("videoId")]
        if not ids:
            return []
        r = c.get(f"{YT}/videos", params={"part": "snippet,statistics,contentDetails", "id": ",".join(ids), "key": key})
        _ok(r, "YouTube")
    out = []
    for v in r.json().get("items", []):
        sn, st = v.get("snippet", {}), v.get("statistics", {})
        out.append({"id": v["id"], "title": sn.get("title", ""), "channel": sn.get("channelTitle", ""),
                    "published": sn.get("publishedAt", ""), "description": sn.get("description", "")[:1500],
                    "views": int(st.get("viewCount", 0) or 0), "likes": int(st.get("likeCount", 0) or 0),
                    "comments": int(st.get("commentCount", 0) or 0), "duration": v.get("contentDetails", {}).get("duration", ""),
                    "url": f"https://www.youtube.com/watch?v={v['id']}"})
    return out


def youtube_comments(video_id: str, n: int = 50) -> list[str]:
    key = _key("YOUTUBE_API_KEY")
    with _client() as c:
        r = c.get(f"{YT}/commentThreads", params={"part": "snippet", "videoId": video_id, "maxResults": min(n, 100),
                                                   "order": "relevance", "textFormat": "plainText", "key": key})
        if r.status_code == 403:
            return []  # comments disabled on this video
        _ok(r, "YouTube")
    return [it["snippet"]["topLevelComment"]["snippet"].get("textDisplay", "") for it in r.json().get("items", [])]


def youtube_own_channel(channel_id: str, videos: int = 20, per_video: int = 50) -> list[dict]:
    """Comments on YOUR channel's latest videos, for persona evidence."""
    key = _key("YOUTUBE_API_KEY")
    with _client() as c:
        r = c.get(f"{YT}/channels", params={"part": "contentDetails", "id": channel_id, "key": key})
        _ok(r, "YouTube")
        items = r.json().get("items", [])
        if not items:
            raise SearchError("No channel found with that id. It starts with UC and is in YouTube Studio, Settings, Channel.")
        uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        r = c.get(f"{YT}/playlistItems", params={"part": "contentDetails,snippet", "playlistId": uploads,
                                                 "maxResults": min(videos, 50), "key": key})
        _ok(r, "YouTube")
        vids = [(it["contentDetails"]["videoId"], it["snippet"].get("title", "")) for it in r.json().get("items", [])]
    out = []
    for vid, title in vids:
        for text in youtube_comments(vid, per_video):
            out.append({"text": text, "source": f"YouTube comment on: {title}"})
    return out


# ---------- Meta Ad Library ----------
def meta_ads(query: str, countries: list[str] | None = None, n: int = 25) -> dict:
    token = _key("META_ACCESS_TOKEN")
    version = os.getenv("META_GRAPH_VERSION", "v23.0")
    import json as _json
    params = {"search_terms": query, "ad_reached_countries": _json.dumps(countries or ["IN"]), "ad_type": "ALL",
              "ad_active_status": "ALL", "limit": min(n, 100), "access_token": token,
              "fields": "page_name,ad_creative_bodies,ad_creative_link_titles,ad_creative_link_descriptions,"
                        "ad_delivery_start_time,ad_delivery_stop_time,ad_snapshot_url,publisher_platforms"}
    with _client() as c:
        r = c.get(f"https://graph.facebook.com/{version}/ads_archive", params=params)
        _ok(r, "Meta Ad Library")
    ads = [{"page": a.get("page_name", ""), "bodies": a.get("ad_creative_bodies", []) or [],
            "titles": a.get("ad_creative_link_titles", []) or [], "start": a.get("ad_delivery_start_time", ""),
            "stop": a.get("ad_delivery_stop_time", ""), "url": a.get("ad_snapshot_url", ""),
            "platforms": a.get("publisher_platforms", []) or []} for a in r.json().get("data", [])]
    return {"ads": ads, "note": "The Ad Library API only returns ads about social issues, elections or politics, "
                                "and ads delivered in the EU. Browse commercial ads at facebook.com/ads/library and paste them in."}


# ---------- Instagram (your own account) ----------
def instagram_own(media: int = 20, per_media: int = 50) -> list[dict]:
    token = _key("INSTAGRAM_ACCESS_TOKEN")
    version = os.getenv("IG_GRAPH_VERSION", "v23.0")
    base = f"https://graph.instagram.com/{version}"
    with _client() as c:
        r = c.get(f"{base}/me/media", params={"fields": "id,caption,permalink,timestamp,like_count,comments_count",
                                              "limit": min(media, 50), "access_token": token})
        _ok(r, "Instagram")
        out = []
        for m in r.json().get("data", []):
            rc = c.get(f"{base}/{m['id']}/comments", params={"fields": "text,timestamp", "limit": min(per_media, 50),
                                                              "access_token": token})
            if rc.status_code >= 400:
                continue
            cap = (m.get("caption") or "")[:60]
            out += [{"text": x.get("text", ""), "source": f"Instagram comment on: {cap}"} for x in rc.json().get("data", [])]
    return out


def _ok(r: httpx.Response, name: str):
    if r.status_code >= 400:
        raise SearchError(f"{name} returned {r.status_code}: {r.text[:300]}")
