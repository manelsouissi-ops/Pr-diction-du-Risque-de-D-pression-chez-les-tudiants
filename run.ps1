# run.ps1 - Start the Flask development server

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Starting Depression Predictor ===" -ForegroundColor Cyan

if (-not (Test-Path "$ProjectDir\.venv\Scripts\Activate.ps1")) {
    Write-Host "Error: Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

& "$ProjectDir\.venv\Scripts\Activate.ps1"

$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"

Write-Host "Starting Flask server at http://127.0.0.1:5000" -ForegroundColor Green
python "$ProjectDir\app.py"
