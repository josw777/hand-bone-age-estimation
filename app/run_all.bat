@echo off
cd /d "%~dp0"
start "BoneAge Backend" cmd /k call "%~dp0run_backend.bat"
timeout /t 3 /nobreak > nul
start "BoneAge Frontend" cmd /k call "%~dp0run_frontend.bat"
