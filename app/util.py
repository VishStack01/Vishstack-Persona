"""Shared helpers: tolerant JSON parsing and PII stripping."""
import json
import re

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.S | re.I)


def parse_json(text: str):
    """Parse a model reply as JSON: whole reply, else one code fence, else first {/[ to last }/]."""
    if text is None:
        raise ValueError("empty reply")
    t = _THINK.sub("", text).strip()  # open reasoning models (Qwen etc.) may emit <think> blocks
    try:
        return json.loads(t)
    except Exception:
        pass
    m = _FENCE.search(t)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if starts:
        s = min(starts)
        e = max(t.rfind("}"), t.rfind("]"))
        if e > s:
            return json.loads(t[s : e + 1])
    raise ValueError("reply was not valid JSON")


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")
_HANDLE = re.compile(r"(^|\s)@[\w.]{2,30}")


def strip_pii(text: str) -> str:
    """Remove emails, phone numbers and @handles. Same rules as the web app."""
    t = _EMAIL.sub("[email removed]", str(text or ""))
    t = _PHONE.sub("[number removed]", t)
    t = _HANDLE.sub(r"\1[handle removed]", t)
    return t.strip()
