@echo off
setlocal

set "APP_DIR=%~dp0"
set "LOCAL_PYTHON=%APP_DIR%.venv\Scripts\pythonw.exe"
set "WORKSPACE_PYTHON=%APP_DIR%..\.venv\Scripts\pythonw.exe"

cd /d "%APP_DIR%"

if exist "%LOCAL_PYTHON%" (
    start "" "%LOCAL_PYTHON%" -m app.main
    exit /b 0
)

if exist "%WORKSPACE_PYTHON%" (
    start "" "%WORKSPACE_PYTHON%" -m app.main
    exit /b 0
)

where pythonw.exe >nul 2>nul
if %errorlevel% equ 0 (
    start "" pythonw.exe -m app.main
    exit /b 0
)

echo Indexator launcher error:
echo No Python interpreter was found.
echo.
echo Checked:
echo   %LOCAL_PYTHON%
echo   %WORKSPACE_PYTHON%
echo   pythonw.exe on PATH
echo.
pause
exit /b 1
