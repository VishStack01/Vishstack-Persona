"""Self-improvement without retraining: DSPy GEPA evolves the panel's instructions from your real results.

It learns from pairs of your posts with real outcomes (data/train/ranker_*.jsonl), reflects on every
wrong prediction, and proposes better instructions. The winner is saved to
data/optimized/panel_instructions.txt, which the app adds to every panel test as "learned guidance".
It is only saved if it beats the current instructions on held-out posts.

  python training/optimize_prompts.py --lm anthropic/claude-sonnet-5-5 --reflection-lm anthropic/claude-opus-5-5
  # open model on vLLM:  --lm openai/Qwen/Qwen3.8-27B --api-base http://localhost:8000/v1
"""
import argparse
import json
import os
from pathlib import Path


def load_pairs(path: Path):
    import dspy
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        m = json.loads(line)["messages"]
        user = m[1]["content"]
        a, b = user.split("\n\nPost B:\n", 1)
        b = b.rsplit("\n\nWhich got", 1)[0]
        rows.append(dspy.Example(post_a=a.replace("Post A:\n", "", 1), post_b=b, winner=m[2]["content"].strip()).with_inputs("post_a", "post_b"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train/ranker_train.jsonl")
    ap.add_argument("--heldout", default="data/train/ranker_heldout.jsonl")
    ap.add_argument("--lm", required=True, help="LiteLLM model string, e.g. anthropic/claude-sonnet-5-5")
    ap.add_argument("--reflection-lm", required=True, help="a strong model, e.g. anthropic/claude-opus-5-5")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--budget", choices=["light", "medium", "heavy"], default="light")
    ap.add_argument("--out", default="data/optimized/panel_instructions.txt")
    a = ap.parse_args()

    import dspy

    class Rank(dspy.Signature):
        """Predict which of two short-form posts gets the better real result with this creator's audience. Answer A or B."""
        post_a: str = dspy.InputField()
        post_b: str = dspy.InputField()
        winner: str = dspy.OutputField(desc="A or B")

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        ok = str(getattr(pred, "winner", "")).strip().upper().startswith(gold.winner)
        fb = "Correct." if ok else (f"Wrong. In real results post {gold.winner} did better. "
                                    "Reflect on what in that post's hook, pacing, trust or call to action this audience responded to.")
        return dspy.Prediction(score=1.0 if ok else 0.0, feedback=fb)

    def acc(program, data):
        if not data:
            return None
        hits = sum(1 for ex in data if metric(ex, program(post_a=ex.post_a, post_b=ex.post_b)).score)
        return hits / len(data)

    kw = {"api_base": a.api_base} if a.api_base else {}
    dspy.configure(lm=dspy.LM(a.lm, **kw))
    train, held = load_pairs(Path(a.train)), load_pairs(Path(a.heldout))
    if len(train) < 20 or not held:
        raise SystemExit(f"Need at least 20 training pairs and some held-out pairs (have {len(train)} and {len(held)}). Log more posts first.")
    split = max(1, int(len(train) * 0.8))
    base = dspy.ChainOfThought(Rank)
    before = acc(base, held)
    gepa = dspy.GEPA(metric=metric, auto=a.budget, reflection_lm=dspy.LM(a.reflection_lm, temperature=1.0, max_tokens=16000))
    tuned = gepa.compile(base, trainset=train[:split], valset=train[split:] or train[:split])
    after = acc(tuned, held)
    print(f"Held-out pairwise accuracy: before {before:.0%}, after {after:.0%}")
    if after is not None and before is not None and after > before:
        text = "\n\n".join(p.signature.instructions for _, p in tuned.named_predictors())
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"Saved learned guidance to {a.out}. The app picks it up on next load.")
    else:
        print("Not saved: the new instructions didn't beat the current ones on held-out posts.")


if __name__ == "__main__":
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    main()
