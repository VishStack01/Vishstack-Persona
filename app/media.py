"""Open-source media tools that run on your machine.

- faster-whisper (Whisper): transcribe interview recordings and videos.
- PySceneDetect: count cuts and measure pacing in a video.

Only derived features are kept (transcript, hook text, cut count, pacing). Uploaded files are deleted
right after analysis. Analyse your own videos, or files you have the right to use.
"""
import os

_model = None


def has_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def has_scenes() -> bool:
    try:
        import scenedetect  # noqa: F401
        return True
    except ImportError:
        return False


def _whisper():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="auto", compute_type="int8")
    return _model


def transcribe(path: str) -> str:
    if not has_whisper():
        raise RuntimeError("Install faster-whisper to transcribe: pip install -r requirements-media.txt")
    segments, _info = _whisper().transcribe(path, beam_size=5, vad_filter=True)
    return "\n".join(s.text.strip() for s in segments if s.text.strip())


def analyze_video(path: str) -> dict:
    """Transcript, the words spoken in the first 3 seconds, number of cuts and average shot length."""
    if not (has_whisper() and has_scenes()):
        raise RuntimeError("Install video analysis first: pip install -r requirements-media.txt")
    from scenedetect import ContentDetector, detect

    segments, info = _whisper().transcribe(path, beam_size=5, vad_filter=True)
    segs = [s for s in segments if s.text.strip()]
    scenes = detect(path, ContentDetector())
    duration = scenes[-1][1].get_seconds() if scenes else (segs[-1].end if segs else 0.0)
    words = sum(len(s.text.split()) for s in segs)
    return {
        "transcript": "\n".join(s.text.strip() for s in segs)[:8000],
        "hook_text": " ".join(s.text.strip() for s in segs if s.start < 3.0),
        "language": getattr(info, "language", "") or "",
        "duration_seconds": round(duration, 1),
        "cuts": max(len(scenes) - 1, 0),
        "avg_shot_seconds": round(duration / len(scenes), 2) if scenes else None,
        "words_per_minute": round(words / (duration / 60), 0) if duration else None,
    }
