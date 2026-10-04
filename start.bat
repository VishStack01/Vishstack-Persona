@echo off
rem One command to run Vishstack Persona on Windows: double-click start.bat
cd /d %~dp0
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if not exist .env (
  copy .env.example .env >nul
  echo Created .env. Add your API keys to it, then run start.bat again.
  pause
  exit /b
)
start "" http://127.0.0.1:8080
uvicorn app.main:app --host 127.0.0.1 --port 8080
