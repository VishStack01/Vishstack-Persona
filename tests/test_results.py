from fastapi.testclient import TestClient

from tests.conftest import jresp


def client():
    from app.main import app
    return TestClient(app)


def test_youtube_results(env, monkeypatch, mock_http):
    monkeypatch.setenv("YOUTUBE_API_KEY", "k")
    mock_http.handler = lambda req: jresp({"items": [{"statistics": {"viewCount": "1200", "likeCount": "80", "commentCount": "9"}}]})
    r = client().post("/api/results", json={"url": "https://youtube.com/shorts/abcdefghijk"}).json()
    assert r == {"platform": "YouTube", "views": 1200, "likes": 80, "comments": 9}
    assert "id=abcdefghijk" in str(mock_http.calls[-1].url)


def test_instagram_results_match_permalink(env, monkeypatch, mock_http):
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "t")
    def h(req):
        if "/insights" in str(req.url):
            return jresp({"data": [{"name": "views", "values": [{"value": 5400}]}, {"name": "reach", "values": [{"value": 3100}]}]})
        return jresp({"data": [{"id": "1", "permalink": "https://www.instagram.com/reel/XYZ/", "like_count": 300, "comments_count": 12}]})
    mock_http.handler = h
    r = client().post("/api/results", json={"url": "https://instagram.com/reel/XYZ?igsh=abc"}).json()
    assert r["likes"] == 300 and r["views"] == 5400 and r["reach"] == 3100


def test_results_rejects_other_sites(env):
    r = client().post("/api/results", json={"url": "https://example.com/post"})
    assert r.status_code == 400
