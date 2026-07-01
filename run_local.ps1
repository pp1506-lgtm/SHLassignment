# Start server (Windows PowerShell)
# Usage: .\run_local.ps1 YOUR_GEMINI_KEY

param([string]$ApiKey)

if (-not $ApiKey) {
    Write-Error "Usage: .\run_local.ps1 YOUR_GEMINI_KEY"
    exit 1
}

$env:GEMINI_API_KEY = $ApiKey
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting SHL Assessment Advisor on http://localhost:8000"
Write-Host "Test: curl http://localhost:8000/health"
Write-Host ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
