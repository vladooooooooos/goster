@echo off
setlocal

cd /d "%~dp0"
echo Starting local Qdrant server...
docker compose up -d qdrant
if errorlevel 1 (
    echo.
    echo Could not start Qdrant server. Make sure Docker Desktop is running.
    echo You can also run this manually:
    echo   docker compose up -d qdrant
    echo.
    pause
    exit /b 1
)

echo.
echo Local Qdrant server start command completed.
echo Qdrant should be available at:
echo   http://127.0.0.1:6333
echo.
pause
