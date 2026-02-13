@echo off
title AI Chatbot Server
cd /d "%~dp0"

echo ============================================================
echo    MEADOWLANDS CHATBOT SERVER
echo ============================================================
echo.

:: Find Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

echo Starting Ngrok tunnel in separate window...
start "Ngrok Tunnel" cmd /k START_TUNNEL.bat

echo.
echo Waiting for tunnel to initialize...
timeout /t 5 /nobreak

echo.
echo Starting FastAPI server...
echo.
python start_chatbot.py

pause
