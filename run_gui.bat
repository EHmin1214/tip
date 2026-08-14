@echo off
chcp 65001 >nul
setlocal
rem ── 나만의 TIP — 인터랙티브 웹앱 실행 (더블클릭) ──────────────────────
rem 구버전은 Sim4Life 9.0 번들 파이썬을 절대경로로 박아둬서 이 머신에서 깨졌다.
rem 이제 사용 가능한 환경을 순서대로 찾는다. Sim4Life 는 GUI 실행에 필요 없다.
set PYTHONIOENCODING=utf-8
set "HERE=%~dp0"
set "APP=%HERE%src\tip\gui\app.py"

echo [tip-gui] 서버 시작 중... 브라우저가 자동으로 열립니다 (http://127.0.0.1:8765)
echo [tip-gui] 종료하려면 이 창에서 Ctrl+C

rem 1순위: 전용 conda 환경 (Sim4Life 비의존 — SETUP.md 권장)
set "PY=%USERPROFILE%\miniconda3\envs\tip\python.exe"
if exist "%PY%" goto :run

rem 2순위: Sim4Life 9.6 번들 기반 venv (numpy 1.26.4 · scipy 1.14.1 포함, 검증됨)
set "PY=%HERE%..\.venv-s4l\Scripts\python.exe"
if exist "%PY%" goto :run

rem 3순위: Sim4Life 9.6 번들 파이썬 직접
set "PY=C:\Program Files\Sim4Life_9.6\Python\python.exe"
if exist "%PY%" goto :run

echo.
echo [tip-gui] 오류: 실행할 파이썬 환경을 찾지 못했습니다.
echo           numpy 와 scipy 가 있는 환경이 필요합니다. 예:
echo             conda create -n tip python=3.11 numpy=1.26.4 scipy matplotlib
echo.
pause
exit /b 1

:run
echo [tip-gui] 사용 환경: %PY%
"%PY%" "%APP%"
pause
