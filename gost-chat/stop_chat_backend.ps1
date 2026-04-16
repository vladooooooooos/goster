$ErrorActionPreference = "Stop"

$hostAddress = "127.0.0.1"
$port = 8000

$connections = Get-NetTCPConnection -LocalAddress $hostAddress -LocalPort $port -State Listen -ErrorAction SilentlyContinue

if (-not $connections) {
    Write-Host "No GOSTer Chat backend is listening at http://$hostAddress`:$port."
    exit 0
}

$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($processId in $processIds) {
    try {
        $process = Get-Process -Id $processId -ErrorAction Stop
        Write-Host "Stopping process $processId ($($process.ProcessName)) listening at http://$hostAddress`:$port."
        Stop-Process -Id $processId -Force -ErrorAction Stop
    }
    catch {
        Write-Warning "Could not stop process $processId. $($_.Exception.Message)"
    }
}
