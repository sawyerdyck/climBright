# stop.ps1 — Stop all climBright services
# Usage: .\stop.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root ".pids"

Write-Host "climBright — Stopping services..." -ForegroundColor Cyan

if (Test-Path $pidFile) {
    $pids = Get-Content $pidFile | Where-Object { $_ -match '^\d+$' }
    foreach ($pid in $pids) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped PID $pid" -ForegroundColor Yellow
        } catch {
            Write-Host "  PID $pid already stopped" -ForegroundColor DarkGray
        }
    }
    Remove-Item $pidFile -Force
} else {
    # Fallback: kill by process name
    Write-Host "  No .pids file found. Killing by name..." -ForegroundColor DarkGray
    Get-Process -Name "mongod" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*main:app*" } | Stop-Process -Force
    Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*server.js*" } | Stop-Process -Force
}

Write-Host "  Done." -ForegroundColor Green
