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

& $pythonPath -c "import torch, torchvision, torchxrayvision" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the local Chest X-ray model (CPU)..." -ForegroundColor Cyan
    & $pythonPath -m pip install --disable-pip-version-check torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
    & $pythonPath -m pip install --disable-pip-version-check -r "backend\requirements-model.txt"
}

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env. Add your API key there when needed." -ForegroundColor Yellow
}

Push-Location "backend"
try { & $pythonPath -m app.tools.download_imaging_model } finally { Pop-Location }

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
