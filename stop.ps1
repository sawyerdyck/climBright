# stop.ps1 - Stop all climBright services
# Usage: powershell -ExecutionPolicy Bypass -File .\stop.ps1

$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }

$pidFile = Join-Path $root ".pids"

Write-Host "climBright - Stopping services..." -ForegroundColor Cyan

if (Test-Path $pidFile) {
    $pids = Get-Content $pidFile | Where-Object { $_ -match '^\d+$' }
    foreach ($procId in $pids) {
        $proc = Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue
        if ($proc) {
            try {
                $proc | Stop-Process -Force -ErrorAction Stop
                Write-Host "  Stopped $($proc.ProcessName) (PID $procId)" -ForegroundColor Yellow
            } catch {
                Write-Host "  Could not stop $($proc.ProcessName) (PID $procId) - may need admin" -ForegroundColor Red
            }
        } else {
            Write-Host "  PID $procId already stopped" -ForegroundColor DarkGray
        }
    }
    Remove-Item $pidFile -Force
    Write-Host "  Done." -ForegroundColor Green
} else {
    Write-Host "  No .pids file found at: $pidFile" -ForegroundColor DarkGray
    Write-Host "  Attempting to kill by name..." -ForegroundColor DarkGray

    $killed = 0
    Get-Process -Name "node" -ErrorAction SilentlyContinue | ForEach-Object {
        try { $_ | Stop-Process -Force -ErrorAction Stop; $killed++; Write-Host "  Stopped node (PID $($_.Id))" -ForegroundColor Yellow } catch {}
    }
    Get-Process -Name "python" -ErrorAction SilentlyContinue | ForEach-Object {
        try { $_ | Stop-Process -Force -ErrorAction Stop; $killed++; Write-Host "  Stopped python (PID $($_.Id))" -ForegroundColor Yellow } catch {}
    }
    Get-Process -Name "mongod" -ErrorAction SilentlyContinue | ForEach-Object {
        try { $_ | Stop-Process -Force -ErrorAction Stop; $killed++; Write-Host "  Stopped mongod (PID $($_.Id))" -ForegroundColor Yellow } catch {
            Write-Host "  Could not stop mongod (PID $($_.Id)) - run as admin or use: Stop-Process -Id $($_.Id) -Force" -ForegroundColor Red
        }
    }

    if ($killed -eq 0) { Write-Host "  No services found running." -ForegroundColor DarkGray }
    Write-Host "  Done." -ForegroundColor Green
}
