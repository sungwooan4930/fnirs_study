@echo off
title fNIRS Monitor

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python 3.10+ from https://www.python.org/
        pause
        exit /b 1
    )
    echo [SETUP] Installing packages...
    .venv\Scripts\pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Package install failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo [DONE] Setup complete.
)

set PYTHONPATH=%~dp0
.venv\Scripts\python src\main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed. See error above.
    pause
)
