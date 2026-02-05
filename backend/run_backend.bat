@echo off
echo Starting Backend...
cd /d "%~dp0"
uvicorn main:app --reload
pause
