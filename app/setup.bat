@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Python virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 goto :error

echo [2/4] Python packages...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
if errorlevel 1 goto :error

echo [3/4] Frontend packages...
cd frontend
call npm install
if errorlevel 1 goto :error
cd ..

echo [4/4] Done.
echo Run run_all.bat to start the dashboard.
pause
exit /b 0

:error
echo.
echo Setup failed. Check Python 3.11, Node.js/npm, CUDA/PyTorch environment.
pause
exit /b 1
