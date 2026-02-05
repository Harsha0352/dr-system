@echo off
echo Starting Model Training...
echo This process may take 30-60 minutes depending on your GPU.
echo.
cd /d "%~dp0"
python model.py
pause
