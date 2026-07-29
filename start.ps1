# start.ps1 - Start all climBright services (MongoDB, FastAPI, Express)
# Usage: powershell -ExecutionPolicy Bypass -File .\start.ps1
# Stop:  powershell -ExecutionPolicy Bypass -File .\stop.ps1

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "climBright - Starting services..." -ForegroundColor Cyan

# --- MongoDB ---
$mongod = $null
$mongoSearch = Get-ChildItem "C:\Program Files\MongoDB\Server" -Recurse -Filter "mongod.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ($mongoSearch) { $mongod = $mongoSearch }
if (-not $mongod) {
    Write-Host "  [!] mongod not found. Install MongoDB." -ForegroundColor Red
    exit 1
}

$dbPath = Join-Path $root "db\mongo"
if (-not (Test-Path $dbPath)) { New-Item -ItemType Directory -Force -Path $dbPath | Out-Null }

Write-Host "  [1/3] MongoDB (port 2701)..." -ForegroundColor Yellow
$mongoProc = Start-Process $mongod -ArgumentList "--dbpath", $dbPath, "--bind_ip", "127.0.0.1", "--port", "2701" -WindowStyle Hidden -PassThru
Write-Host "        PID: $($mongoProc.Id)" -ForegroundColor DarkGray

Start-Sleep -Seconds 2

# --- FastAPI (AI model server) ---
Write-Host "  [2/3] FastAPI (port 9000)..." -ForegroundColor Yellow
$fastapiProc = Start-Process python -ArgumentList "-m", "uvicorn", "main:app", "--port", "9000" -WorkingDirectory $root -WindowStyle Hidden -PassThru
Write-Host "        PID: $($fastapiProc.Id)" -ForegroundColor DarkGray

Start-Sleep -Seconds 3

# --- Express (frontend) ---
$nodeExe = $null
if (Test-Path "C:\Program Files\nodejs\node.exe") {
    $nodeExe = "C:\Program Files\nodejs\node.exe"
} else {
    $found = Get-Command node -ErrorAction SilentlyContinue
    if ($found) { $nodeExe = $found.Source }
}
if (-not $nodeExe) {
    Write-Host "  [!] node not found. Install Node.js." -ForegroundColor Red
    exit 1
}

$frontendDir = Join-Path $root "frontend"

# Install deps if needed
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "        Installing npm dependencies..." -ForegroundColor DarkGray
    $npmCmd = Join-Path (Split-Path $nodeExe) "npm.cmd"
    & $npmCmd install --prefix $frontendDir 2>$null
}

Write-Host "  [3/3] Express (port 3000)..." -ForegroundColor Yellow
$expressProc = Start-Process $nodeExe -ArgumentList "server.js" -WorkingDirectory $frontendDir -WindowStyle Hidden -PassThru
Write-Host "        PID: $($expressProc.Id)" -ForegroundColor DarkGray

Start-Sleep -Seconds 2

# --- Health check ---
try {
    $health = Invoke-WebRequest -Uri http://127.0.0.1:3000/health -UseBasicParsing -TimeoutSec 5
    if ($health.StatusCode -eq 200) {
        Write-Host ""
        Write-Host "  All services running!" -ForegroundColor Green
        Write-Host "  App:     http://127.0.0.1:3000" -ForegroundColor White
        Write-Host "  FastAPI: http://127.0.0.1:9000/docs" -ForegroundColor White
        Write-Host ""
        Write-Host "  Run .\stop.ps1 to shut everything down." -ForegroundColor DarkGray
    }
} catch {
    Write-Host "  [!] Express health check failed. Check frontend/.env and logs." -ForegroundColor Red
}

# Save PIDs for stop script
$pidFile = Join-Path $root ".pids"
@($mongoProc.Id, $fastapiProc.Id, $expressProc.Id) | Out-File $pidFile -Encoding ascii
