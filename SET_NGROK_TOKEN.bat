@echo off
title Set Ngrok Authtoken
cd /d "%~dp0"

echo ============================================================
echo    NGROK AUTHENTICATION SETUP
echo ============================================================
echo.
echo Ngrok requires a FREE account for public URLs.
echo.
echo 1. Sign up at: https://dashboard.ngrok.com/signup
echo 2. Get your authtoken at: https://dashboard.ngrok.com/get-started/your-authtoken
echo.
set /p NGROK_TOKEN="Paste your ngrok authtoken here: "

if "%NGROK_TOKEN%"=="" (
    echo.
    echo [ERROR] No token entered!
    pause
    exit /b 1
)

echo.
echo Setting up ngrok authtoken...
python -c "from pyngrok import ngrok; ngrok.set_auth_token('%NGROK_TOKEN%'); print('[OK] Ngrok authtoken saved!')"

echo.
echo Done! You can now run START_SERVER.bat
echo.
pause
