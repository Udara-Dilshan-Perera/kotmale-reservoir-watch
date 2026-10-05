@echo off
cd /d %~dp0

REM ---- VIVA DEMONSTRATION SETTING ------------------------------------------
REM The historical data ends on 2025-12-31, so the demo treats 2026-01-01 as "today".
REM For a real deployment, delete the next line so the computer's own date is used.
set KOTMALE_SYSTEM_DATE=2026-01-01

REM Optional: require a PIN to add or correct records.
REM set KOTMALE_ENTRY_PIN=1234

start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8000"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
