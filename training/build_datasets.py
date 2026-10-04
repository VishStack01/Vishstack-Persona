"""Build training sets from YOUR real data. Run on any machine.

Reads data/workspace/*.json (written by the app) and, optionally, a survey CSV.
Writes JSONL chat files to data/train/:

  ranker_train.jsonl / ranker_heldout.jsonl   which of two real posts did better (time split: newest 20% held out)
  writer_sft.jsonl                            your best-performing real posts, to teach your voice and structure
  persona_sft.jsonl                           survey answers in respondents' own words (SubPOP / Stanford-style)

Nothing here invents labels: every example comes from a real post result or a real survey answer.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.util import strip_pii  # noqa: E402

PII_COLS = {"name", "email", "phone", "mobile", "handle", "username", "id", "user_id", "ip"}


def load_ws(d: Path) -> dict:
    ws = {}
    for k in ("core", "variants", "posts", "facts"):
        f = d / f"{k}.json"
        ws[k] = json.loads(f.read_text()) if f.exists() else {}
    return ws


def content(v: dict) -> str:
    beats = "\n".join(f"{b.get('t','')} {b.get('line','')}" for b in v.get("beats", []))
    return f"Hook: {v.get('hook','')}\nOn screen: {v.get('onScreen','')}\nScript:\n{beats}\nCaption: {v.get('caption','')}\nCall to action: {v.get('cta','')}"


def metric_value(p: dict, metric: str):
    a = p.get("actual") or {}
    if not a.get("views"):
        return None
    return a.get("hold", 0) if metric == "hold" else a.get("actions", 0) / a["views"]


def build_ranker(ws: dict, metric: str, seed: int):
    variants = {v["id"]: v for v in ws["variants"].get("items", [])}
    posts = [p for p in ws["posts"].get("items", []) if metric_value(p, metric) is not None and p["variantId"] in variants]
    posts.sort(key=lambda p: p["lockedAt"])
    cut = max(1, int(len(posts) * 0.8)) if len(posts) >= 5 else len(posts)
    platform = ws["core"].get("brief", {}).get("platform", "short-form video")
    label = "hold rate (share of viewers who kept watching past the first seconds)" if metric == "hold" else "actions per view"
    rnd = random.Random(seed)

    def pairs(group):
        out = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                va, vb = metric_value(a, metric), metric_value(b, metric)
                if va == vb:
                    continue
                if rnd.random() < 0.5:
                    a, b, va, vb = b, a, vb, va
                out.append({"messages": [
                    {"role": "system", "content": f"You predict which of two {platform} posts gets the higher {label} with this creator's real audience. Answer with A or B only."},
                    {"role": "user", "content": f"Post A:\n{content(variants[a['variantId']])}\n\nPost B:\n{content(variants[b['variantId']])}\n\nWhich got the higher {label}?"},
                    {"role": "assistant", "content": "A" if va > vb else "B"}]})
        return out

    return pairs(posts[:cut]), pairs(posts[cut:]) if cut < len(posts) else []


def build_writer(ws: dict, metric: str):
    variants = {v["id"]: v for v in ws["variants"].get("items", [])}
    posts = [p for p in ws["posts"].get("items", []) if metric_value(p, metric) is not None and p["variantId"] in variants]
    if len(posts) < 8:
        return []
    posts.sort(key=lambda p: metric_value(p, metric), reverse=True)
    top = posts[: max(2, len(posts) // 4)]
    brief = ws["core"].get("brief", {})
    out = []
    for p in top:
        v = variants[p["variantId"]]
        out.append({"messages": [
            {"role": "system", "content": "You write short-form content in this creator's voice. State no fact unless it is given with a source tag like [F1]."},
            {"role": "user", "content": f"Platform: {brief.get('platform','')}. Language: {brief.get('language','')}. Angle: {v.get('angle','')}. Write the post."},
            {"role": "assistant", "content": content(v)}]})
    return out


def build_persona(survey: Path):
    if not survey or not survey.exists():
        return []
    out = []
    with survey.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q, a = (row.get("question") or "").strip(), strip_pii(row.get("answer") or "")
            if not q or len(a) < 2:
                continue
            profile = "; ".join(f"{k}: {v}" for k, v in row.items()
                                if k not in ("question", "answer") and k.lower() not in PII_COLS and v)
            out.append({"messages": [
                {"role": "system", "content": "You answer as a real member of this creator's audience, matching how such a person actually answers."},
                {"role": "user", "content": f"Respondent profile: {profile or 'not given'}\nQuestion: {q}"},
                {"role": "assistant", "content": a}]})
    return out


def write(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default="data/workspace")
    ap.add_argument("--survey", default="data/survey_responses.csv", help="CSV with question, answer and profile columns")
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--metric", choices=["hold", "actions"], default="hold")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    ws = load_ws(Path(a.workspace))
    train, held = build_ranker(ws, a.metric, a.seed)
    writer = build_writer(ws, a.metric)
    persona = build_persona(Path(a.survey))
    out = Path(a.out)
    write(out / "ranker_train.jsonl", train)
    write(out / "ranker_heldout.jsonl", held)
    write(out / "writer_sft.jsonl", writer)
    write(out / "persona_sft.jsonl", persona)
    print(f"ranker: {len(train)} train pairs, {len(held)} held-out pairs")
    print(f"writer: {len(writer)} examples from your best posts")
    print(f"persona: {len(persona)} survey answers")
    if len(train) < 500:
        print("Note: under ~500 ranker pairs a fine-tune will mostly memorise. Keep using prompting, GEPA and calibration until you have more posts.")
    if len(persona) < 2000:
        print("Note: persona fine-tunes need thousands of real answers (the SubPOP study used 70K). Keep collecting consented survey data.")
    return {"ranker_train": len(train), "ranker_heldout": len(held), "writer": len(writer), "persona": len(persona)}


if __name__ == "__main__":
    main()
