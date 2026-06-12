<#
.SYNOPSIS
    Start the FastAPI backend and Next.js frontend in separate windows.
.EXAMPLE
    .\scripts\start-dev.ps1
#>
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "Restoring calibration to committed values (a=1.8, b=1.8, threshold=0.6)..." -ForegroundColor Cyan
git -C $ROOT restore api/app/state/model_state.json

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ROOT\api'; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
)

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ROOT\frontend'; npm run dev"
)

Write-Host ""
Write-Host "Started backend and frontend in new windows." -ForegroundColor Green
Write-Host "  Dashboard : http://localhost:3000"
Write-Host "  API docs  : http://localhost:8000/docs"
Write-Host "  Health    : http://localhost:8000/health"
