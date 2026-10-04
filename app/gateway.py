"""One gateway for every model: Claude, GPT, Gemini, and open models served by vLLM, SGLang or Ollama.

Models and roles live in config/models.yaml. A model is used only if it has a model id,
is not disabled, and (for hosted providers) its API key is set.
"""
import json
import os
import time
from pathlib import Path

import httpx
import yaml

from .settings import DATA, ROOT
from .util import parse_json

CONFIG_PATH = Path(os.getenv("VSP_MODELS", ROOT / "config" / "models.yaml"))
TIMEOUT = float(os.getenv("VSP_MODEL_TIMEOUT", "300"))
TRANSPORT = None  # tests replace this with httpx.MockTransport


class ModelError(Exception):
    pass


def load_models():
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    models = {}
    for m in cfg.get("models", []):
        key = os.getenv(m["api_key_env"]) if m.get("api_key_env") else None
        hosted = m["provider"] in ("anthropic", "openai", "google")
        enabled = bool(m.get("enabled", True)) and bool(m.get("model")) and bool(key or not hosted)
        models[m["name"]] = {**m, "api_key": key, "enabled": enabled}
    return models, cfg.get("roles", {})


def resolve(tier: str = "default", explicit: str | None = None) -> dict:
    models, roles = load_models()
    for name in (explicit, roles.get(tier), roles.get("default")):
        if name and name in models and models[name]["enabled"]:
            return models[name]
    for m in models.values():
        if m["enabled"]:
            return m
    raise ModelError("No model is ready. Add an API key to .env or enable a local model in config/models.yaml.")


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, transport=TRANSPORT)


def call_text(m: dict, prompt: str, max_tokens: int = 8000) -> str:
    p = m["provider"]
    with _client() as c:
        if p == "anthropic":
            r = c.post("https://api.anthropic.com/v1/messages",
                       headers={"x-api-key": m["api_key"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                       json={"model": m["model"], "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
            _raise(r, m)
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
        if p in ("openai", "openai_compatible"):
            base = "https://api.openai.com/v1" if p == "openai" else m["base_url"].rstrip("/")
            body = {"model": m["model"], "messages": [{"role": "user", "content": prompt}]}
            # OpenAI's current models take max_completion_tokens; vLLM/Ollama/OpenRouter take max_tokens
            body["max_completion_tokens" if p == "openai" else "max_tokens"] = max_tokens
            headers = {"Authorization": f"Bearer {m['api_key']}"} if m.get("api_key") else {}
            r = c.post(f"{base}/chat/completions", headers=headers, json=body)
            _raise(r, m)
            return r.json()["choices"][0]["message"].get("content") or ""
        if p == "google":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m['model']}:generateContent"
            r = c.post(url, params={"key": m["api_key"]},
                       json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                             "generationConfig": {"maxOutputTokens": max_tokens}})
            _raise(r, m)
            cands = r.json().get("candidates") or []
            parts = cands[0].get("content", {}).get("parts", []) if cands else []
            return "".join(x.get("text", "") for x in parts)
    raise ModelError(f"Unknown provider {p!r} for model {m['name']}")


def _raise(r: httpx.Response, m: dict):
    if r.status_code >= 400:
        raise ModelError(f"{m['name']} returned {r.status_code}: {r.text[:300]}")


def _log(m: dict, ms: int, prompt: str, reply: str, ok: bool):
    try:
        with open(DATA / "model_calls.jsonl", "a") as f:
            f.write(json.dumps({"at": time.time(), "model": m["name"], "ms": ms, "prompt_chars": len(prompt),
                                "reply_chars": len(reply or ""), "ok": ok}) + "\n")
    except OSError:
        pass


def calls_last_24h() -> int:
    f = DATA / "model_calls.jsonl"
    if not f.exists():
        return 0
    since = time.time() - 86400
    n = 0
    for line in f.read_text().splitlines()[-20000:]:
        try:
            if json.loads(line).get("at", 0) >= since:
                n += 1
        except ValueError:
            continue
    return n


def call_json(prompt: str, tier: str = "default", model: str | None = None):
    """Ask a model for JSON. Retries once with a format reminder. Returns (data, model_name)."""
    cap = int(os.getenv("VSP_MAX_CALLS_PER_DAY", "400"))
    if calls_last_24h() >= cap:
        raise ModelError(f"Daily limit of {cap} model calls reached, to protect your API bill. Raise VSP_MAX_CALLS_PER_DAY in .env to continue.")
    m = resolve(tier, model)
    t0 = time.time()
    reply = call_text(m, prompt)
    try:
        data = parse_json(reply)
    except ValueError:
        reply = call_text(m, prompt + "\n\nYour previous reply was not valid JSON. Reply with only the JSON, nothing else.")
        try:
            data = parse_json(reply)
        except ValueError as e:
            _log(m, int((time.time() - t0) * 1000), prompt, reply, False)
            raise ModelError(f"{m['name']} did not return valid JSON. Try again or use less data.") from e
    _log(m, int((time.time() - t0) * 1000), prompt, reply, True)
    return data, m["name"]
