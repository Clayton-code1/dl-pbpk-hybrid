<#
.SYNOPSIS
    Create venv, install deps, and train the hybrid DL-PK model on Theophylline data.
.EXAMPLE
    .\scripts\train_hybrid_theoph.ps1
#>
$ErrorActionPreference = "Stop"

$ROOT    = Split-Path -Parent $PSScriptRoot          # repo root (dl-pbpk-hybrid)
$VENV    = Join-Path $ROOT "src\.venv"
$VPYTHON = Join-Path $VENV "Scripts\python.exe"
$REQ     = Join-Path $ROOT "src\requirements.txt"
$TRAIN   = Join-Path $ROOT "src\training\train_hybrid_theoph.py"

Write-Host "=== Hybrid DL-PK Training Pipeline ===" -ForegroundColor Cyan

# --- Create venv if missing ---
if (-not (Test-Path $VPYTHON)) {
    Write-Host "`n[1/3] Creating venv at $VENV ..."
    py -3.11 -m venv "$VENV"
} else {
    Write-Host "`n[1/3] Venv already exists at $VENV"
}

# --- Install dependencies ---
Write-Host "[2/3] Installing dependencies ..."
& $VPYTHON -m pip install --quiet --upgrade pip
& $VPYTHON -m pip install --quiet -r $REQ

# --- Train ---
Write-Host "[3/3] Starting training ...`n"
& $VPYTHON -u $TRAIN

Write-Host "`n=== Done ===" -ForegroundColor Green
