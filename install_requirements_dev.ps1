$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $rootDir "requirements-dev.txt"
$venvPythonPath = Join-Path $rootDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "Missing requirements file: $requirementsPath"
}

if (Test-Path -LiteralPath $venvPythonPath) {
    $pythonExe = $venvPythonPath
}
else {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Could not find python.exe. Create .venv or add Python to PATH, then run this shortcut again."
    }
    $pythonExe = $pythonCommand.Source
}

Write-Host "Installing dependencies from:"
Write-Host "  $requirementsPath"
Write-Host ""
Write-Host "Using Python:"
Write-Host "  $pythonExe"
Write-Host ""

& $pythonExe -m pip install -r $requirementsPath

Write-Host ""
Write-Host "Dependency installation finished."
Write-Host "Press Enter to close this window."
[void][System.Console]::ReadLine()
