param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "Creating the local Python environment..." -ForegroundColor Cyan
    python -m venv $venvPath
    & $pythonPath -m pip install --disable-pip-version-check --upgrade pip
    & $pythonPath -m pip install --disable-pip-version-check -r "backend\requirements.txt"
}

if (-not (Test-Path -LiteralPath "frontend\node_modules")) {
    Write-Host "Installing the web interface..." -ForegroundColor Cyan
    Push-Location "frontend"
    try { npm ci } finally { Pop-Location }
}

Write-Host "Building the web interface..." -ForegroundColor Cyan
Push-Location "frontend"
try { npm run build } finally { Pop-Location }

Write-Host ""
Write-Host "PULSELINE is ready at http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop it." -ForegroundColor DarkGray
Set-Location -LiteralPath (Join-Path $projectRoot "backend")
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port
