@echo off
chcp 65001 >nul
title fNIRS 집중도 모니터

:: ── 가상환경 확인 및 자동 설치 ────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [설치] 가상환경을 처음 설정합니다. 잠시 기다려주세요...
    python -m venv .venv
    if errorlevel 1 (
        echo [오류] Python이 설치되어 있지 않습니다.
        echo   https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치해주세요.
        pause
        exit /b 1
    )
    echo [설치] 필요한 패키지를 설치합니다...
    .venv\Scripts\pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [오류] 패키지 설치 실패. 인터넷 연결을 확인해주세요.
        pause
        exit /b 1
    )
    echo [완료] 설치 완료.
)

:: ── 앱 실행 ───────────────────────────────────────────────
set PYTHONPATH=%~dp0
.venv\Scripts\python src\main.py
if errorlevel 1 (
    echo.
    echo [오류] 앱 실행 중 문제가 발생했습니다. 위 오류 메시지를 확인해주세요.
    pause
)
