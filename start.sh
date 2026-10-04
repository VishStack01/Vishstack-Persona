#!/usr/bin/env bash
# One command to run Vishstack Persona on Mac or Linux: ./start.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; echo "Created .env. Add your API keys to it, then run ./start.sh again."; exit 0; fi
( sleep 2; (command -v open >/dev/null && open http://127.0.0.1:8080) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8080) || true ) &
exec uvicorn app.main:app --host 127.0.0.1 --port 8080
