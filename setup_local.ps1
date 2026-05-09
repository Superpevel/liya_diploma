# Logo Generation ML - Windows local setup (Python 3.11 only)
# Run from project folder:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_local.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ParentDir = Split-Path -Parent $ScriptDir

# Require Python 3.11 (project deps are pinned to versions tested on 3.11).
$PythonCmd = $null
$PythonArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $PythonCmd = "py"
        $PythonArgs = @("-3.11")
    }
}
if (-not $PythonCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $v = & python --version 2>&1
    if ($v -match "Python 3\.11") { $PythonCmd = "python" }
}
if (-not $PythonCmd -and (Get-Command python3.11 -ErrorAction SilentlyContinue)) {
    $PythonCmd = "python3.11"
}
if (-not $PythonCmd) {
    Write-Host "ERROR: Python 3.11 not found." -ForegroundColor Red
    Write-Host "Install with:  winget install Python.Python.3.11"
    Write-Host "Or download:   https://www.python.org/downloads/release/python-3119/"
    Write-Host "Then restart PowerShell and re-run this script."
    exit 1
}

Write-Host "=== Logo Generation ML - Local Setup (Windows) ===" -ForegroundColor Cyan
Write-Host "Project: $ScriptDir"
$resolvedVersion = & $PythonCmd @PythonArgs --version 2>&1
Write-Host "Using: $PythonCmd $($PythonArgs -join ' ') -> $resolvedVersion" -ForegroundColor DarkGray

# Step 0: create / reuse a project venv (.venv311) so we don't pollute system Python.
$VenvDir = Join-Path $ScriptDir ".venv311"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host ""
    Write-Host "Step 0: Creating venv at $VenvDir ..." -ForegroundColor Yellow
    & $PythonCmd @PythonArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host ""
    Write-Host "Step 0: Reusing existing venv at $VenvDir" -ForegroundColor DarkGray
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

# Step 1: PyTorch (CUDA 12.8 if NVIDIA GPU present, else CPU)
Write-Host ""
Write-Host "Step 1: Installing PyTorch 2.6+..." -ForegroundColor Yellow

$hasNvidia = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi *> $null
    if ($LASTEXITCODE -eq 0) { $hasNvidia = $true }
}

if ($hasNvidia) {
    Write-Host "NVIDIA GPU detected - installing CUDA 12.8 build"
    & $VenvPython -m pip install --upgrade "torch>=2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "No NVIDIA GPU detected - installing CPU build (LoRA training will not run locally)"
    & $VenvPython -m pip install --upgrade "torch>=2.6.0" torchvision
}
if ($LASTEXITCODE -ne 0) { throw "pip install torch failed" }

# Step 2: project dependencies
Write-Host ""
Write-Host "Step 2: Installing project dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install -r "$ScriptDir\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }

# Step 3: ai-toolkit (needed for LoRA training)
Write-Host ""
Write-Host "Step 3: Setting up ai-toolkit..." -ForegroundColor Yellow
$AiToolkit = Join-Path $ParentDir "ai-toolkit"
if (-not (Test-Path $AiToolkit)) {
    & git clone https://github.com/ostris/ai-toolkit $AiToolkit
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    Write-Host "ai-toolkit cloned to $AiToolkit"
} else {
    Write-Host "ai-toolkit already exists: $AiToolkit"
}
# Always (re)install ai-toolkit deps — pip is idempotent, and a previous run
# may have skipped this if the folder already existed.
& $VenvPython -m pip install -r "$AiToolkit\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install ai-toolkit requirements failed" }

# Step 4: smoke check
Write-Host ""
Write-Host "Step 4: Smoke-testing imports..." -ForegroundColor Yellow
& $VenvPython -c "import torch, diffusers, transformers, accelerate, bitsandbytes; from resvg_py import svg_to_bytes; from PIL import Image; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Activate venv: .\.venv311\Scripts\Activate.ps1"
Write-Host "Run notebooks: .\.venv311\Scripts\python.exe -m jupyter lab"
Write-Host "ai-toolkit at: $AiToolkit"
Write-Host ""
Write-Host "HuggingFace auth (needed for FLUX.1-dev):"
Write-Host "  .\.venv311\Scripts\huggingface-cli.exe login"
