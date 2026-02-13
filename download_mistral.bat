@echo off
echo Installing Ollama Mistral model...
echo.
echo Please make sure Ollama is installed from: https://ollama.ai/download/windows
echo.
echo Attempting to pull Mistral model...
echo.

REM Try different possible Ollama paths
if exist "C:\Program Files\Ollama\ollama.exe" (
    echo Found Ollama at C:\Program Files\Ollama
    "C:\Program Files\Ollama\ollama.exe" pull mistral
    goto :done
)

if exist "C:\Program Files (x86)\Ollama\ollama.exe" (
    echo Found Ollama at C:\Program Files (x86)\Ollama
    "C:\Program Files (x86)\Ollama\ollama.exe" pull mistral
    goto :done
)

if exist "%LOCALAPPDATA%\Ollama\ollama.exe" (
    echo Found Ollama at %LOCALAPPDATA%\Ollama
    "%LOCALAPPDATA%\Ollama\ollama.exe" pull mistral
    goto :done
)

echo.
echo Could not find Ollama executable. 
echo Please ensure Ollama is properly installed from https://ollama.ai/download/windows
echo After installation, run this script again.
pause
goto :end

:done
echo.
echo Mistral model download complete!
echo You can now run: python ai_main.py
pause

:end
