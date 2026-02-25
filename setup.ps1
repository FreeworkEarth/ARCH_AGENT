# ARCH_AGENT Setup — Windows (PowerShell)
# Run with: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== ARCH_AGENT Setup ===" -ForegroundColor Cyan

# Python dependencies
Write-Host ""
Write-Host "[1/3] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Ollama
Write-Host ""
Write-Host "[2/3] Checking Ollama..." -ForegroundColor Yellow
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "  Ollama not found. Downloading installer..."
    $installerUrl = "https://ollama.com/download/OllamaSetup.exe"
    $installerPath = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    Write-Host "  Running installer (follow the prompts)..."
    Start-Process -FilePath $installerPath -Wait
    Write-Host "  Ollama installed. Please restart PowerShell if 'ollama' is not found after this step."
} else {
    Write-Host "  Ollama already installed: $(ollama --version)"
}

# Models
Write-Host ""
Write-Host "[3/3] Pulling required models (this may take several minutes)..." -ForegroundColor Yellow
Write-Host "  Pulling deepseek-r1:14b (~9 GB)..."
ollama pull deepseek-r1:14b

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Quick start:"
Write-Host "  cd 01_stage_analyze"
Write-Host '  python LLM_frontend_upgraded.py "analyze https://github.com/apache/commons-io.git all-time 5 timesteps with deepseek-r1:14b and answer: how did the architecture evolve?"'
