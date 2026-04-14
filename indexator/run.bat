@echo off
setlocal
cd /d "%~dp0indexator"
".\.venv\Scripts\python.exe" -m app.main
pause
