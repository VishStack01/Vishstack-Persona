import base64
import json
import time

from fastapi.testclient import TestClient


def client():
    from app.main import app
    return TestClient(app)


def test_password_gate(env, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    c = client()
    assert c.get("/api/models").status_code == 401
    good = base64.b64encode(b"me:s3cret").decode()
    assert c.get("/api/models", headers={"Authorization": f"Basic {good}"}).status_code == 200
    bad = base64.b64encode(b"me:nope").decode()
    assert c.get("/api/models", headers={"Authorization": f"Basic {bad}"}).status_code == 401


def test_no_password_by_default(env, monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert client().get("/api/models").status_code == 200


def test_daily_cap_blocks_calls(env, monkeypatch):
    from app import gateway, settings
    monkeypatch.setattr(gateway, "DATA", env)
    monkeypatch.setenv("VSP_MAX_CALLS_PER_DAY", "2")
    with open(env / "model_calls.jsonl", "w") as f:
        for _ in range(2):
            f.write(json.dumps({"at": time.time(), "model": "x"}) + "\n")
    try:
        gateway.call_json("hi")
        assert False, "should have been blocked"
    except gateway.ModelError as e:
        assert "Daily limit" in str(e)


def test_analyze_video_without_tools_explains(env, monkeypatch):
    from app import media
    monkeypatch.setattr(media, "has_whisper", lambda: False)
    r = client().post("/api/analyze-video", files={"file": ("a.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400
    assert "requirements-media.txt" in r.json()["detail"]
