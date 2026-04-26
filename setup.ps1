# setup.ps1 - Create venv and install dependencies

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Depression Predictor Setup ===" -ForegroundColor Cyan

if (-not (Test-Path "$ProjectDir\.venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv "$ProjectDir\.venv"
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$ProjectDir\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel

Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r "$ProjectDir\requirements.txt"

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate venv:  .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Run notebooks:  jupyter nbconvert --to notebook --execute notebooks\01_EDA.ipynb --output notebooks\01_EDA.ipynb"
Write-Host "                     jupyter nbconvert --to notebook --execute notebooks\02_Modeling.ipynb --output notebooks\02_Modeling.ipynb"
Write-Host "  3. Start app:      python app.py"
Write-Host "  4. Or use run.ps1: .\run.ps1"
