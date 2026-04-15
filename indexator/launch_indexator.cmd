@echo off
setlocal

set "APP_DIR=C:\goster\indexator"
set "LOCAL_PYTHON=C:\goster\indexator\.venv\Scripts\pythonw.exe"
set "FALLBACK_PYTHON=C:\goster\.venv\Scripts\pythonw.exe"

cd /d "%APP_DIR%"

if exist "%LOCAL_PYTHON%" (
    start "" "%LOCAL_PYTHON%" -m app.main
    exit /b 0
)

if exist "%FALLBACK_PYTHON%" (
    start "" "%FALLBACK_PYTHON%" -m app.main
    exit /b 0
)

echo Indexator launcher error:
echo Neither Python interpreter was found.
echo.
echo Checked:
echo   %LOCAL_PYTHON%
echo   %FALLBACK_PYTHON%
echo.
pause
exit /b 1
