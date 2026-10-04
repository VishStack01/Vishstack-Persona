# Vishstack Persona

Test content on 1,000 personas built from your real audience before you post, then prove the panel works against real results.
It creates content, searches what already exists, checks every claim, simulates your audience, refines the weakest part,
and learns from your real post results.

## What's inside

| Job | Tools used | Where |
|---|---|---|
| Models | Claude, GPT, Gemini, plus open models (Qwen3.8-27B, Sarvam-30B, Gemma 4, any OpenRouter model) through one gateway | `app/gateway.py`, `config/models.yaml` |
| Serving open models | vLLM, SGLang or Ollama (OpenAI-compatible endpoints) | `docker-compose.yml` |
| Search existing content | YouTube Data API (public views, likes, comments), Brave or Tavily web search, Meta Ad Library API, polite page fetch with robots.txt check and trafilatura | `app/search.py` |
| Your own accounts | Instagram API (your media and comments), your YouTube channel's comments | `app/search.py` |
| Interviews and videos | faster-whisper (open-source Whisper) transcribes recordings; PySceneDetect measures a video's hook, cuts and pacing, all on your machine | `app/media.py` |
| Personas | Evidence-backed segments weighted to 1,000 personas; every trait cites real evidence | `web/index.html` |
| Truth check | Claims checked only against your fact library; quotes must be found word for word | `web/index.html`, `evals/` |
| Multi-model jury | Second opinion from a different model family; disagreement is shown as uncertainty | `roles.second_opinion` |
| Autopilot | One click from brief to a tested, refined winner, with every rule still enforced | `web/index.html` |
| Real results | Fetches views, likes and comments for your own YouTube and Instagram posts | `app/results.py` |
| Self-improvement | DSPy GEPA evolves the panel instructions from your real results | `training/optimize_prompts.py` |
| Fine-tuning | Unsloth LoRA/QLoRA on Qwen3.8-27B, Sarvam-30B or Gemma 4 | `training/train_lora.py` |
| Safety gate | No new model or prompt goes live unless it beats the current one and passes zero false claims | `training/eval_gate.py` |

## Run it (laptop, 10 minutes)

Fastest: run `./start.sh` (Mac or Linux) or double-click `start.bat` (Windows). The first run creates `.env`;
add your keys, run it again, and the app opens in your browser. Or do it by hand:


```bash
git clone https://github.com/VishStack01/Vishstack-Persona.git && cd Vishstack-Persona
# copy this project's files in, then:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add at least ANTHROPIC_API_KEY (or enable a local model)
uvicorn app.main:app --host 127.0.0.1 --port 8080
```
Open http://127.0.0.1:8080. The app saves your workspace to `data/workspace/` on your machine.

Turn on more sources by adding keys to `.env`: `YOUTUBE_API_KEY`, `BRAVE_API_KEY` or `TAVILY_API_KEY`,
`META_ACCESS_TOKEN`, `INSTAGRAM_ACCESS_TOKEN`. For transcription and video analysis: `pip install -r requirements-media.txt`.
Fill in the GPT and Gemini model ids in `config/models.yaml` to use them.

## Put it in your GitHub repo

```bash
git clone https://github.com/VishStack01/Vishstack-Persona.git
cd Vishstack-Persona
# unzip this project here so README.md sits at the repo root, then:
git add . && git status          # check: no .env, no data/ files listed
git commit -m "Vishstack Persona v1" && git push
```
`.env` and everything in `data/` (your audience data and workspace) are in `.gitignore` and never get pushed.

## Running it anywhere but your own laptop

Set `APP_PASSWORD` in `.env` first. Without it, anyone who can reach the app can read your audience data and spend your API credits.
`VSP_MAX_CALLS_PER_DAY` (default 400) stops model calls for the day once the limit is reached.

## Open models on a GPU

```bash
docker compose --profile gpu up        # app + vLLM serving Qwen3.8-27B on port 8000
```
Then set `enabled: true` for `qwen` in `config/models.yaml`. For Hinglish-heavy audiences also try `sarvam`.
On a laptop, use Ollama with a Gemma 4 or small Qwen build and the `gemma-ollama` entry.

## The learning loop

1. Use the app for every post: lock the panel's prediction and your gut score in Calibrate, add real results after 7 days.
2. `python training/build_datasets.py` turns your real results and survey answers into training files.
3. `python training/optimize_prompts.py --lm anthropic/claude-sonnet-5-5 --reflection-lm anthropic/claude-opus-5-5`
   (needs about 20+ post pairs). Better instructions are saved only if they beat the old ones on held-out posts, and the app uses them automatically.
4. With hundreds of posts or thousands of survey answers: `python training/train_lora.py ...` on a GPU (see `training/README.md`).
5. `python training/eval_gate.py --candidate vsp-tuned --baseline claude-sonnet` decides if the new model may be used.

## Autopilot

In Create, **Run autopilot** does the whole loop in one click: write versions, truth-check every claim, fix what fails,
test on your 1,000 personas, get a second opinion from another model, refine the winner's biggest leak, re-check and
re-test. You watch each step and approve the result; nothing is posted.

## Real results without typing

When you lock a prediction, add the post's link. After it's live, **Fetch results** pulls views, likes and comments
from YouTube, or likes, comments and (when your account allows) views and reach from Instagram. Hold rate and
downloads or purchases aren't in those APIs; add them from YouTube Studio, Instagram insights or your app console.
The scoreboard can compare on hold rate, actions per view, views, or likes and comments per view.

## Rules this code enforces

- Personas come only from data you're allowed to use. Handles, emails and phone numbers are stripped.
- Traits without evidence are dropped; segments with fewer than 3 evidence items are excluded.
- A claim is supported only if its quote is found word for word in your fact library. False claims block a draft.
- Panel answers that don't cite the segment's own traits are left out and counted.
- Refining stops after 3 loops, in autopilot too. Nothing posts automatically.
- Model calls stop for the day at `VSP_MAX_CALLS_PER_DAY`, and the top bar shows how many you've used.
- Search uses official APIs and robots.txt. Nothing behind a login is scraped. The Meta Ad Library API only covers
  political, issue and EU ads; browse other ads at facebook.com/ads/library and paste them in.
- Before offering this to other businesses, get legal advice on data use (India's DPDP Act) and each platform's terms.

## Tests

```bash
pip install pytest && python -m pytest -q
```
