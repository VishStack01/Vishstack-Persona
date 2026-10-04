"""Vishstack Persona server: serves the web app and the API it calls.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8080
Then open http://127.0.0.1:8080
"""
import base64
import json
import os
import secrets
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import gateway, results, search
from .settings import DATA, WEB

app = FastAPI(title="Vishstack Persona")
WS_KEYS = {"core", "evidence", "references", "facts", "variants", "posts"}
WS_DIR = DATA / "workspace"
WS_DIR.mkdir(parents=True, exist_ok=True)
OVERRIDES = DATA / "optimized" / "panel_instructions.txt"
MAX_UPLOAD = 500 * 1024 * 1024


@app.middleware("http")
async def password_gate(request: Request, call_next):
    """Set APP_PASSWORD in .env whenever the app is reachable from anywhere but your own machine."""
    pw = os.getenv("APP_PASSWORD", "")
    if pw:
        ok = False
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                ok = secrets.compare_digest(base64.b64decode(header[6:]).decode().partition(":")[2], pw)
            except Exception:
                ok = False
        if not ok:
            return JSONResponse({"detail": "Password required."}, status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="Vishstack Persona"'})
    return await call_next(request)


async def _save_upload(file: UploadFile, default: str) -> str:
    suffix = Path(file.filename or default).suffix or Path(default).suffix
    size = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(413, "That file is over 500 MB.")
            tmp.write(chunk)
        return tmp.name


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/models")
def models():
    ms, roles = gateway.load_models()
    return {"models": [{"name": k, "provider": v["provider"], "model": v.get("model", ""), "enabled": v["enabled"]}
                       for k, v in ms.items()],
            "roles": roles, "sources": search.available(),
            "learned": OVERRIDES.read_text().strip() if OVERRIDES.exists() else ""}


class AskIn(BaseModel):
    prompt: str = Field(max_length=400_000)
    tier: str = "default"
    model: str | None = None


@app.post("/api/ask")
def ask(body: AskIn):
    try:
        data, used = gateway.call_json(body.prompt, body.tier, body.model)
    except gateway.ModelError as e:
        raise HTTPException(502, str(e))
    return {"data": data, "model": used}


# ---------- Workspace (one JSON file per section, read by the training scripts) ----------
@app.get("/api/workspace")
def ws_all():
    out = {}
    for k in WS_KEYS:
        f = WS_DIR / f"{k}.json"
        if f.exists():
            out[k] = json.loads(f.read_text())
    return out


@app.put("/api/workspace/{key}")
def ws_put(key: str, body: dict):
    if key not in WS_KEYS:
        raise HTTPException(404, "Unknown workspace section.")
    tmp = WS_DIR / f".{key}.tmp"
    tmp.write_text(json.dumps(body.get("v", body), ensure_ascii=False))
    tmp.replace(WS_DIR / f"{key}.json")
    return {"ok": True}


# ---------- Search ----------
class Q(BaseModel):
    query: str
    n: int = 15
    region: str = "IN"
    language: str | None = None
    published_after: str | None = None
    video_duration: str | None = None
    countries: list[str] | None = None


def _wrap(fn, *a, **k):
    try:
        return fn(*a, **k)
    except search.SearchError as e:
        raise HTTPException(400, str(e))


@app.post("/api/search/web")
def s_web(q: Q):
    return {"items": _wrap(search.web_search, q.query, q.n)}


@app.post("/api/search/youtube")
def s_yt(q: Q):
    return {"items": _wrap(search.youtube_search, q.query, q.n, q.region, q.language, q.published_after, q.video_duration)}


class VidIn(BaseModel):
    video_id: str
    n: int = 50


@app.post("/api/search/youtube/comments")
def s_ytc(b: VidIn):
    return {"items": _wrap(search.youtube_comments, b.video_id, b.n)}


@app.post("/api/search/meta-ads")
def s_meta(q: Q):
    return _wrap(search.meta_ads, q.query, q.countries or [q.region], q.n)


class UrlIn(BaseModel):
    url: str


@app.post("/api/fetch")
def s_fetch(b: UrlIn):
    return _wrap(search.fetch_page, b.url)


# ---------- Real results for your own posts, and usage ----------
@app.post("/api/results")
def post_results(b: "UrlIn"):
    return _wrap(results.fetch, b.url)


@app.get("/api/usage")
def usage():
    return {"calls_24h": gateway.calls_last_24h(), "cap": int(os.getenv("VSP_MAX_CALLS_PER_DAY", "400"))}


# ---------- Your own accounts ----------
class ChannelIn(BaseModel):
    channel_id: str
    videos: int = 20
    per_video: int = 50


@app.post("/api/own/youtube")
def own_yt(b: ChannelIn):
    return {"items": _wrap(search.youtube_own_channel, b.channel_id, b.videos, b.per_video)}


class IgIn(BaseModel):
    media: int = 20
    per_media: int = 50


@app.post("/api/own/instagram")
def own_ig(b: IgIn):
    return {"items": _wrap(search.instagram_own, b.media, b.per_media)}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    from . import media
    path = await _save_upload(file, "audio.bin")
    try:
        text = media.transcribe(path)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(path)
    return {"text": text}


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    from . import media
    path = await _save_upload(file, "video.mp4")
    try:
        return media.analyze_video(path)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Couldn't analyse that file: {e}")
    finally:
        os.unlink(path)  # the video itself is never kept
