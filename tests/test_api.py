import httpx
from fastapi.testclient import TestClient

from tests.conftest import jresp


def client(env):
    import importlib
    import app.settings, app.main
    importlib.reload(app.settings)
    importlib.reload(app.main)
    return TestClient(app.main.app)


def test_models_and_workspace_roundtrip(env):
    c = client(env)
    j = c.get("/api/models").json()
    assert {m["name"] for m in j["models"]} >= {"claude-sonnet", "qwen", "sarvam"}
    assert j["sources"]["youtube"] is False and j["sources"]["fetch"] is True
    assert c.put("/api/workspace/evidence", json={"v": {"items": [{"id": "E1", "text": "hi"}]}}).json() == {"ok": True}
    assert c.get("/api/workspace").json()["evidence"]["items"][0]["id"] == "E1"
    assert c.put("/api/workspace/secrets", json={"v": {}}).status_code == 404
    assert c.get("/").status_code == 200


def test_ask_without_models_gives_clear_error(env):
    r = client(env).post("/api/ask", json={"prompt": "hi"})
    assert r.status_code == 502 and "No model is ready" in r.json()["detail"]


def test_search_needs_keys(env):
    c = client(env)
    r = c.post("/api/search/youtube", json={"query": "cibil"})
    assert r.status_code == 400 and "YOUTUBE_API_KEY" in r.json()["detail"]
    assert "BRAVE_API_KEY" in c.post("/api/search/web", json={"query": "x"}).json()["detail"]


def test_youtube_search_parses(env, monkeypatch, mock_http):
    monkeypatch.setenv("YOUTUBE_API_KEY", "k")

    def handler(req):
        if req.url.path.endswith("/search"):
            return jresp({"items": [{"id": {"videoId": "abc"}}]})
        if req.url.path.endswith("/videos"):
            return jresp({"items": [{"id": "abc", "snippet": {"title": "CIBIL tips", "channelTitle": "Ch", "publishedAt": "2026-09-01"},
                                     "statistics": {"viewCount": "120000", "likeCount": "5000", "commentCount": "300"},
                                     "contentDetails": {"duration": "PT45S"}}]})
        raise AssertionError(req.url)

    mock_http.handler = handler
    items = client(env).post("/api/search/youtube", json={"query": "cibil"}).json()["items"]
    assert items[0]["views"] == 120000 and items[0]["url"].endswith("abc")


def test_fetch_respects_robots(env, mock_http):
    def handler(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private")
        return httpx.Response(200, text="<html><head><title>Plans</title></head><body><article><p>" + "Premium costs Rs 199 a month. " * 20 + "</p></article></body></html>")

    mock_http.handler = handler
    c = client(env)
    assert c.post("/api/fetch", json={"url": "https://example.com/private/x"}).status_code == 400
    j = c.post("/api/fetch", json={"url": "https://example.com/pricing"}).json()
    assert "Rs 199" in j["text"]


def test_meta_ads_note(env, monkeypatch, mock_http):
    monkeypatch.setenv("META_ACCESS_TOKEN", "t")
    mock_http.handler = lambda req: jresp({"data": [{"page_name": "P", "ad_creative_bodies": ["Vote"], "ad_delivery_start_time": "2026-09-01"}]})
    j = client(env).post("/api/search/meta-ads", json={"query": "x"}).json()
    assert j["ads"][0]["page"] == "P" and "EU" in j["note"]
