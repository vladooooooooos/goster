@echo off
setlocal

cd /d "%~dp0"
echo Stopping local Qdrant server...
docker compose stop qdrant
if errorlevel 1 (
    echo.
    echo Could not stop Qdrant server. Make sure Docker Desktop is running.
    echo You can also run this manually:
    echo   docker compose stop qdrant
    echo.
    pause
    exit /b 1
)

echo.
echo Local Qdrant server stopped.
echo.
pause
