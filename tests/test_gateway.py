import json

import pytest

from app import gateway
from tests.conftest import jresp


def test_resolve_skips_models_without_keys(env, monkeypatch):
    with pytest.raises(gateway.ModelError):
        gateway.resolve("default")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert gateway.resolve("default")["name"] == "claude-sonnet"
    assert gateway.resolve("complex")["name"] == "claude-opus"
    # second_opinion points at gpt, which has no id/key, so it falls back to default
    assert gateway.resolve("second_opinion")["name"] == "claude-sonnet"
    assert gateway.resolve("default", explicit="claude-haiku")["name"] == "claude-haiku"


def test_provider_shapes(env, monkeypatch, mock_http):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def handler(req):
        if "anthropic" in req.url.host:
            assert req.headers["x-api-key"] == "k"
            return jresp({"content": [{"type": "text", "text": '{"ok":"claude"}'}]})
        if req.url.path.endswith("/chat/completions"):
            body = json.loads(req.content)
            assert "max_tokens" in body  # openai_compatible uses max_tokens
            return jresp({"choices": [{"message": {"content": '<think>hmm</think>{"ok":"qwen"}'}}]})
        if "generativelanguage" in req.url.host:
            return jresp({"candidates": [{"content": {"parts": [{"text": '{"ok":"gemini"}'}]}}]})
        raise AssertionError(req.url)

    mock_http.handler = handler
    assert gateway.call_json("x")[0] == {"ok": "claude"}
    qwen = {"name": "qwen", "provider": "openai_compatible", "base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen3.8-27B"}
    from app.util import parse_json
    assert parse_json(gateway.call_text(qwen, "x")) == {"ok": "qwen"}
    gem = {"name": "gemini", "provider": "google", "model": "g", "api_key": "k"}
    assert parse_json(gateway.call_text(gem, "x")) == {"ok": "gemini"}


def test_retry_on_bad_json_then_error(env, monkeypatch, mock_http):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    replies = iter(["not json", '{"fixed":true}'])
    mock_http.handler = lambda req: jresp({"content": [{"type": "text", "text": next(replies)}]})
    assert gateway.call_json("x")[0] == {"fixed": True}
    mock_http.handler = lambda req: jresp({"content": [{"type": "text", "text": "still not json"}]})
    with pytest.raises(gateway.ModelError):
        gateway.call_json("x")


def test_http_error_is_readable(env, monkeypatch, mock_http):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    mock_http.handler = lambda req: jresp({"error": "overloaded"}, 529)
    with pytest.raises(gateway.ModelError, match="529"):
        gateway.call_json("x")
