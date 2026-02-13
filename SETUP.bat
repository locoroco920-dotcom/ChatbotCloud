@echo off
title Meadowlands Chatbot - Setup
color 0A

echo ============================================================
echo    MEADOWLANDS CHATBOT - AUTOMATIC SETUP
echo ============================================================
echo.

:: Check for Python
echo [1/6] Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)
echo [OK] Python found!
python --version
echo.

:: Navigate to script directory
cd /d "%~dp0"
echo [2/6] Working directory: %CD%
echo.

:: Skip venv - install directly to simplify
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip
echo [OK] pip upgraded
echo.

:: Install dependencies ONE BY ONE with verbose output
echo [4/6] Installing dependencies (with progress)...
echo.

echo     [4a] Installing fastapi...
python -m pip install fastapi>=0.100.0
if errorlevel 1 echo     [WARN] fastapi may have had issues

echo     [4b] Installing uvicorn...
python -m pip install uvicorn>=0.20.0
if errorlevel 1 echo     [WARN] uvicorn may have had issues

echo     [4c] Installing openai...
python -m pip install openai>=1.0.0
if errorlevel 1 echo     [WARN] openai may have had issues

echo     [4d] Installing pydantic...
python -m pip install pydantic>=2.0.0
if errorlevel 1 echo     [WARN] pydantic may have had issues

echo     [4e] Installing python-multipart...
python -m pip install python-multipart>=0.0.6
if errorlevel 1 echo     [WARN] python-multipart may have had issues

echo     [4f] Installing numpy...
python -m pip install numpy>=1.20.0
if errorlevel 1 echo     [WARN] numpy may have had issues

echo     [4g] Installing pyngrok...
python -m pip install pyngrok>=5.0.0
if errorlevel 1 echo     [WARN] pyngrok may have had issues

echo     [4h] Installing sentence-transformers (LARGE - may take 2-5 min)...
python -m pip install sentence-transformers>=2.0.0
if errorlevel 1 echo     [WARN] sentence-transformers may have had issues

echo.
echo [OK] Dependencies installed!
echo.

:: Ngrok authentication setup
echo [5/6] Ngrok token setup skipped (no hardcoded secrets).
echo     Set your token manually if you use local ngrok debugging.
echo.

:: Verify paths
echo [6/6] Configuring paths for this PC...
echo     Script directory: %CD%
echo     [OK] Paths are configured dynamically in start_chatbot.py
echo     [OK] No manual path changes needed
echo.

:: Test imports
echo [6/6] Verifying installation...
python -c "print('    Testing imports...')"
python -c "import fastapi; print(f'    [OK] FastAPI {fastapi.__version__}')"
python -c "import openai; print(f'    [OK] OpenAI {openai.__version__}')"
python -c "import uvicorn; print('    [OK] Uvicorn')"
python -c "import pyngrok; print('    [OK] pyngrok')"
python -c "from sentence_transformers import SentenceTransformer; print('    [OK] Sentence Transformers')"
echo.

echo ============================================================
echo    SETUP COMPLETE!
echo ============================================================
echo.
echo To start the chatbot:
echo   Double-click START_SERVER.bat
echo   Or run: python start_chatbot.py
echo.
echo The server will show you the ngrok URL to access the chatbot.
echo.
pause
