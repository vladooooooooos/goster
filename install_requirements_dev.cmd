@echo off
setlocal

powershell.exe -ExecutionPolicy Bypass -File "%~dp0install_requirements_dev.ps1"

endlocal
