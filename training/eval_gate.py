"""The gate every new model or checkpoint must pass before the app may use it.

Scores models from config/models.yaml on:
  1. truth: evals/truth_gate.jsonl, where letting a false or unsupported claim pass is an automatic fail
  2. ranking: data/train/ranker_heldout.jsonl, pairwise accuracy on your real posts

  python training/eval_gate.py --candidate vsp-tuned --baseline claude-sonnet
Exit code 0 = promote, 1 = keep the baseline.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import gateway  # noqa: E402
from app.util import _THINK, parse_json  # noqa: E402

TRUTH_PROMPT = """You are a strict fact checker. Using ONLY the fact library, label the claim:
"supported" (a library entry states it), "contradicted" (a library entry says otherwise) or "unverifiable".
Never use outside knowledge. Reply with only JSON: {{"verdict": "..."}}

Fact library:
{facts}

Claim: {claim}"""


def truth_score(model_name: str, path: Path):
    m = gateway.resolve(explicit=model_name)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    correct = leaks = 0
    for r in rows:
        prompt = TRUTH_PROMPT.format(facts="\n".join(f"[F{i+1}] {f}" for i, f in enumerate(r["facts"])), claim=r["claim"])
        try:
            verdict = str(parse_json(gateway.call_text(m, prompt, 400)).get("verdict", "")).lower()
        except Exception:
            verdict = "error"
        correct += verdict == r["expected"]
        leaks += verdict == "supported" and r["expected"] != "supported"
    return correct / len(rows), leaks


def rank_score(model_name: str, path: Path):
    if not path.exists():
        return None
    m = gateway.resolve(explicit=model_name)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        return None
    hits = 0
    for r in rows:
        msgs = r["messages"]
        reply = _THINK.sub("", gateway.call_text(m, msgs[0]["content"] + "\n\n" + msgs[1]["content"], 2000)).strip().upper()
        hits += reply[:1] == msgs[2]["content"]
    return hits / len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--truth", default="evals/truth_gate.jsonl")
    ap.add_argument("--heldout", default="data/train/ranker_heldout.jsonl")
    ap.add_argument("--margin", type=float, default=0.02)
    a = ap.parse_args(argv)
    res = {}
    for name in (a.baseline, a.candidate):
        t, leaks = truth_score(name, Path(a.truth))
        res[name] = {"truth": t, "leaks": leaks, "rank": rank_score(name, Path(a.heldout))}
        r = res[name]["rank"]
        print(f"{name:>16}  truth {t:.0%}  false claims passed {leaks}  ranking {'n/a' if r is None else f'{r:.0%}'}")
    c, b = res[a.candidate], res[a.baseline]
    if c["leaks"] > 0:
        print("FAIL: the candidate let a false or unsupported claim pass.")
        return 1
    if c["truth"] + 1e-9 < b["truth"]:
        print("FAIL: the candidate is worse at fact checking than the baseline.")
        return 1
    if c["rank"] is not None and b["rank"] is not None and c["rank"] < b["rank"] + a.margin:
        print(f"FAIL: ranking isn't at least {a.margin:.0%} better than the baseline.")
        return 1
    print("PASS: promote the candidate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
