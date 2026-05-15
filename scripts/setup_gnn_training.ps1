<#
.SYNOPSIS
    Create a clean Python 3.11 venv for GNN training and validate the environment.
.DESCRIPTION
    - Creates (or recreates) src\.venv with py -3.11
    - Installs pinned dependencies from src\requirements.txt
    - Validates NumPy (must be less than 2), RDKit, PyTorch, and Pandas
.PARAMETER Recreate
    Delete existing venv and start fresh.
.EXAMPLE
    .\scripts\setup_gnn_training.ps1
    .\scripts\setup_gnn_training.ps1 -Recreate
#>

[CmdletBinding()]
param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$ROOT    = Split-Path -Parent $PSScriptRoot
$VENV    = Join-Path $ROOT "src\.venv"
$VPYTHON = Join-Path $VENV "Scripts\python.exe"
$REQ     = Join-Path $ROOT "src\requirements.txt"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " GNN Training Environment Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# -- Recreate if requested ------------------------------------------------

if ($Recreate -and (Test-Path $VENV)) {
    Write-Host "[*] Removing existing venv at $VENV ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VENV
}

# -- Create venv -----------------------------------------------------------

if (-not (Test-Path $VPYTHON)) {
    Write-Host "[1/3] Creating venv with Python 3.11 ..."
    try {
        py -3.11 -m venv "$VENV"
    }
    catch {
        Write-Host "ERROR: py -3.11 not found. Install Python 3.11 from python.org" -ForegroundColor Red
        Write-Host "       and ensure the 'py' launcher is on PATH." -ForegroundColor Red
        exit 1
    }
    Write-Host "      Venv created at $VENV" -ForegroundColor Green
}
else {
    Write-Host "[1/3] Venv already exists at $VENV" -ForegroundColor Green
}

# -- Install dependencies --------------------------------------------------

if (-not (Test-Path $REQ)) {
    Write-Host "ERROR: Requirements file not found: $REQ" -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Upgrading pip and installing dependencies ..."
& $VPYTHON -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip upgrade failed" -ForegroundColor Red
    exit 1
}

& $VPYTHON -m pip install --quiet -r $REQ
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed. Check src\requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host "      Dependencies installed" -ForegroundColor Green

# -- Validation checks -----------------------------------------------------

Write-Host "[3/3] Validating environment ...`n"

$allGood = $true

# NumPy
$npVersion = & $VPYTHON -c "import numpy as np; print(np.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    if ([int]($npVersion.Split('.')[0]) -ge 2) {
        Write-Host "  FAIL  NumPy $npVersion (>= 2.x) - need less than 2" -ForegroundColor Red
        $allGood = $false
    }
    else {
        Write-Host "  OK    NumPy $npVersion" -ForegroundColor Green
    }
}
else {
    Write-Host "  FAIL  NumPy import failed" -ForegroundColor Red
    $allGood = $false
}

# RDKit
$rdkitOk = & $VPYTHON -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO') is not None)" 2>&1
if ($LASTEXITCODE -eq 0 -and $rdkitOk -eq "True") {
    Write-Host "  OK    RDKit (SMILES parse works)" -ForegroundColor Green
}
else {
    Write-Host "  FAIL  RDKit import or parse failed" -ForegroundColor Red
    Write-Host "        Try: pip install rdkit-pypi" -ForegroundColor Yellow
    $allGood = $false
}

# PyTorch
$torchVersion = & $VPYTHON -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK    PyTorch $torchVersion" -ForegroundColor Green
}
else {
    Write-Host "  FAIL  PyTorch import failed" -ForegroundColor Red
    Write-Host "        Try: pip install 'torch>=2.2'" -ForegroundColor Yellow
    $allGood = $false
}

# Pandas
$pandasOk = & $VPYTHON -c "import pandas; print(pandas.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK    Pandas $pandasOk" -ForegroundColor Green
}
else {
    Write-Host "  FAIL  Pandas import failed" -ForegroundColor Red
    Write-Host "        Try: pip install 'pandas>=2.0'" -ForegroundColor Yellow
    $allGood = $false
}

Write-Host ""

if ($allGood) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Environment ready!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
}
else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " Some checks failed - see above." -ForegroundColor Red
    Write-Host " Run with -Recreate to start fresh:" -ForegroundColor Red
    Write-Host "   .\scripts\setup_gnn_training.ps1 -Recreate" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Red
    exit 1
}
