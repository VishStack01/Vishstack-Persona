import json
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VSP_DATA", str(tmp_path))
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "YOUTUBE_API_KEY", "BRAVE_API_KEY",
              "TAVILY_API_KEY", "META_ACCESS_TOKEN", "INSTAGRAM_ACCESS_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


@pytest.fixture
def mock_http(monkeypatch):
    """Route all outgoing HTTP to a handler you set: mock_http.handler = fn(request) -> httpx.Response."""
    from app import gateway

    class H:
        handler = None
        calls = []

    def dispatch(req):
        H.calls.append(req)
        return H.handler(req)

    monkeypatch.setattr(gateway, "TRANSPORT", httpx.MockTransport(dispatch))
    return H


def jresp(obj, status=200):
    return httpx.Response(status, content=json.dumps(obj).encode(), headers={"content-type": "application/json"})
