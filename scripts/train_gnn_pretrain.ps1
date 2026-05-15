<#
.SYNOPSIS
    Activate the GNN training venv and run ADME pretraining.
.DESCRIPTION
    - Calls setup_gnn_training.ps1 if the venv is missing
    - Activates src\.venv
    - Runs pretrain_gnn_adme.py, forwarding any extra arguments
.EXAMPLE
    .\scripts\train_gnn_pretrain.ps1
    .\scripts\train_gnn_pretrain.ps1 --max-samples 500 --rebuild-cache
#>

$ErrorActionPreference = "Stop"

$ROOT    = Split-Path -Parent $PSScriptRoot
$VENV    = Join-Path $ROOT "src\.venv"
$VPYTHON = Join-Path $VENV "Scripts\python.exe"
$SETUP   = Join-Path $ROOT "scripts\setup_gnn_training.ps1"
$TRAIN   = Join-Path $ROOT "src\training\pretrain_gnn_adme.py"

Write-Host "`n=== GNN ADME Pretraining ===" -ForegroundColor Cyan

# -- Ensure venv exists ----------------------------------------------------

if (-not (Test-Path $VPYTHON)) {
    Write-Host "Venv not found - running setup ...`n" -ForegroundColor Yellow
    & $SETUP
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# -- Run pretraining -------------------------------------------------------

Write-Host "Starting pretraining ...`n"
& $VPYTHON -u $TRAIN @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPretraining failed (exit code $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=== Done ===" -ForegroundColor Green
