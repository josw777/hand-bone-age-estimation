@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
