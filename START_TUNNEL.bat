@echo off
title Ngrok Tunnel
cd /d "%~dp0"

echo ============================================================
echo    NGROK TUNNEL
echo ============================================================
echo.
echo Starting tunnel to localhost:8000...
echo.

python -c "from pyngrok import ngrok; url = ngrok.connect(8000); print(); print('========================================'); print('PUBLIC URL:', url); print('========================================'); print(); print('Share this URL to access the chatbot!'); print(); input('Press Enter to stop tunnel...')"

pause
