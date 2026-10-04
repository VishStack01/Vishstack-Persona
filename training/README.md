# Training

| Script | Needs | What it does |
|---|---|---|
| `build_datasets.py` | any machine | Builds JSONL training files from your real results and survey answers |
| `optimize_prompts.py` | API keys | DSPy GEPA: improves panel instructions from your real outcomes, saves only if better |
| `train_lora.py` | NVIDIA GPU, 24 GB+ for 27B QLoRA | Unsloth LoRA fine-tune of Qwen3.8-27B, Sarvam-30B, Gemma 4 or a small Qwen for the ranker |
| `eval_gate.py` | the model served | Promotes a model only if it beats the baseline and lets zero false claims through |

Install on the GPU machine: `pip install -r requirements-train.txt` (follow Unsloth's install guide for your CUDA version).

Survey CSV format for persona training (`data/survey_responses.csv`): columns `question`, `answer`, plus any profile
columns such as `age_band`, `city_tier`, `occupation`. Columns named name, email, phone, handle or id are dropped.

How much data before fine-tuning is worth it: a ranker needs hundreds of post pairs; a persona model needs thousands of
real answers (the SubPOP study used 70,000). Before that, GEPA plus calibration gives you more for less.

`train_lora.py` and `optimize_prompts.py` were written against the current Unsloth, TRL and DSPy 3 APIs but could not be
run on a GPU or real models in the environment they were built in. Run a small test (`--epochs 0.1`) first.
