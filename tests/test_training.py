import json

from training import build_datasets, eval_gate


def make_ws(d, n=10):
    (d / "workspace").mkdir()
    vs = [{"id": f"V{i}", "hook": f"hook {i}", "beats": [{"t": "0-2s", "line": "x"}], "caption": "c", "cta": "d", "angle": "a"} for i in range(n)]
    posts = [{"id": f"P{i}", "variantId": f"V{i}", "lockedAt": i, "gut": 5, "predicted": {"stop": 100, "act": 5},
              "actual": {"views": 1000, "hold": 10 + i, "actions": i}} for i in range(n)]
    (d / "workspace/core.json").write_text(json.dumps({"brief": {"platform": "Instagram Reels", "language": "Hinglish"}}))
    (d / "workspace/variants.json").write_text(json.dumps({"items": vs}))
    (d / "workspace/posts.json").write_text(json.dumps({"items": posts}))
    (d / "survey.csv").write_text("question,answer,age_band,email\nWhy do you follow?,Tips help me @friend,25-34,a@b.com\n")


def test_build_datasets(tmp_path):
    make_ws(tmp_path)
    counts = build_datasets.main(["--workspace", str(tmp_path / "workspace"), "--survey", str(tmp_path / "survey.csv"), "--out", str(tmp_path / "train")])
    assert counts["ranker_train"] == 28 and counts["ranker_heldout"] == 1  # 8 train posts -> 28 pairs, 2 held out -> 1 pair
    assert counts["writer"] == 2 and counts["persona"] == 1
    row = json.loads((tmp_path / "train/persona_sft.jsonl").read_text())
    text = json.dumps(row)
    assert "a@b.com" not in text and "@friend" not in text and "25-34" in text
    held = json.loads((tmp_path / "train/ranker_heldout.jsonl").read_text())
    assert held["messages"][2]["content"] in ("A", "B")


def test_eval_gate_blocks_false_claim_leaks(tmp_path, monkeypatch):
    from app import gateway
    truth = tmp_path / "t.jsonl"
    truth.write_text(json.dumps({"facts": ["Rs 199 a month"], "claim": "free forever", "expected": "contradicted"}) + "\n")
    monkeypatch.setattr(gateway, "resolve", lambda tier="default", explicit=None: {"name": explicit})
    monkeypatch.setattr(gateway, "call_text", lambda m, p, n=0: '{"verdict":"contradicted"}' if m["name"] == "base" else '{"verdict":"supported"}')
    assert eval_gate.main(["--candidate", "cand", "--baseline", "base", "--truth", str(truth), "--heldout", str(tmp_path / "none.jsonl")]) == 1
    monkeypatch.setattr(gateway, "call_text", lambda m, p, n=0: '{"verdict":"contradicted"}')
    assert eval_gate.main(["--candidate", "cand", "--baseline", "base", "--truth", str(truth), "--heldout", str(tmp_path / "none.jsonl")]) == 0
