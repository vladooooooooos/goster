@echo off
setlocal

powershell.exe -ExecutionPolicy Bypass -File "%~dp0stop_chat_backend.ps1"

endlocal
