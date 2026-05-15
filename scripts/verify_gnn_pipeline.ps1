<#
.SYNOPSIS
    Verify that all GNN training artifacts are present and ready for API use.
.DESCRIPTION
    Checks for pretrained GNN encoder, fine-tuned hybrid model, and required
    support files. Prints a clear summary of what is ready and what is missing.
.EXAMPLE
    .\scripts\verify_gnn_pipeline.ps1
#>

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot

$PRETRAIN_DIR  = Join-Path $ROOT "artifacts\models\gnn_pretrain_v1"
$FINETUNE_DIR  = Join-Path $ROOT "artifacts\models\hybrid_gnn_pbpk_theoph_v1"
$MLP_DIR       = Join-Path $ROOT "artifacts\models\hybrid_theoph_v1"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " GNN Pipeline Verification" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$allReady = $true

# -- 1. Pretrained GNN encoder --------------------------------------------

Write-Host "--- Pretrained GNN Encoder ---" -ForegroundColor Yellow
Write-Host "    Directory: $PRETRAIN_DIR"

$pretrainFiles = @("model_gnn.pt", "metrics.json")
$pretrainOk = $true

foreach ($f in $pretrainFiles) {
    $fPath = Join-Path $PRETRAIN_DIR $f
    if (Test-Path $fPath) {
        $size = (Get-Item $fPath).Length
        Write-Host "    OK    $f ($size bytes)" -ForegroundColor Green
    }
    else {
        Write-Host "    MISS  $f" -ForegroundColor Red
        $pretrainOk = $false
    }
}

$scalerPath = Join-Path $PRETRAIN_DIR "scaler.json"
if (Test-Path $scalerPath) {
    Write-Host "    OK    scaler.json" -ForegroundColor Green
}

if ($pretrainOk) {
    Write-Host "    Status: READY" -ForegroundColor Green
}
else {
    Write-Host "    Status: MISSING - run pretraining first" -ForegroundColor Red
    $allReady = $false
}

Write-Host ""

# -- 2. Fine-tuned hybrid GNN-PBPK model ----------------------------------

Write-Host "--- Fine-tuned Hybrid GNN-PBPK Model ---" -ForegroundColor Yellow
Write-Host "    Directory: $FINETUNE_DIR"

$finetuneFiles = @("model.pt", "config.json", "metrics.json", "scaler.json")
$finetuneOk = $true

foreach ($f in $finetuneFiles) {
    $fPath = Join-Path $FINETUNE_DIR $f
    if (Test-Path $fPath) {
        $size = (Get-Item $fPath).Length
        Write-Host "    OK    $f ($size bytes)" -ForegroundColor Green
    }
    else {
        Write-Host "    MISS  $f" -ForegroundColor Red
        $finetuneOk = $false
    }
}

if ($finetuneOk) {
    Write-Host "    Status: READY" -ForegroundColor Green
}
else {
    Write-Host "    Status: MISSING - run fine-tuning first" -ForegroundColor Red
    $allReady = $false
}

Write-Host ""

# -- 3. MLP fallback model ------------------------------------------------

Write-Host "--- MLP Fallback Model ---" -ForegroundColor Yellow
Write-Host "    Directory: $MLP_DIR"

if (Test-Path (Join-Path $MLP_DIR "model.pt")) {
    Write-Host "    OK    MLP fallback available" -ForegroundColor Green
}
else {
    Write-Host "    WARN  MLP fallback not found (non-blocking)" -ForegroundColor Yellow
}

Write-Host ""

# -- 4. Metrics summary ---------------------------------------------------

if ($pretrainOk) {
    $metricsPath = Join-Path $PRETRAIN_DIR "metrics.json"
    $m = Get-Content $metricsPath | ConvertFrom-Json
    Write-Host "--- Pretrain Metrics ---" -ForegroundColor Yellow
    Write-Host "    Train RMSE : $($m.train.rmse)"
    Write-Host "    Val   RMSE : $($m.val.rmse)"
    Write-Host "    Epochs     : $($m.n_epochs)"
    Write-Host ""
}

if ($finetuneOk) {
    $metricsPath = Join-Path $FINETUNE_DIR "metrics.json"
    $m = Get-Content $metricsPath | ConvertFrom-Json
    Write-Host "--- Fine-tune Metrics ---" -ForegroundColor Yellow
    Write-Host "    Train RMSE : $($m.train.rmse) mg/L"
    Write-Host "    Val   RMSE : $($m.val.rmse) mg/L"
    Write-Host "    Epochs     : $($m.n_epochs)"
    Write-Host ""
}

# -- 5. Final verdict -----------------------------------------------------

Write-Host "========================================" -ForegroundColor Cyan

if ($allReady) {
    Write-Host " Pretrained model  : FOUND" -ForegroundColor Green
    Write-Host " Finetuned model   : FOUND" -ForegroundColor Green
    Write-Host " Ready for API     : YES" -ForegroundColor Green
    Write-Host ""
    Write-Host " The API will use model_used='gnn' automatically." -ForegroundColor Green
    Write-Host " Restart the API to pick up the new artifacts:" -ForegroundColor Cyan
    Write-Host "   docker-compose restart api" -ForegroundColor White
    Write-Host "   # or: uvicorn app.main:app --reload  (from api/)" -ForegroundColor White
}
else {
    Write-Host " Pretrained model  : $(if ($pretrainOk) {'FOUND'} else {'MISSING'})" -ForegroundColor $(if ($pretrainOk) {'Green'} else {'Red'})
    Write-Host " Finetuned model   : $(if ($finetuneOk) {'FOUND'} else {'MISSING'})" -ForegroundColor $(if ($finetuneOk) {'Green'} else {'Red'})
    Write-Host " Ready for API     : NO" -ForegroundColor Red
    Write-Host ""
    if (-not $pretrainOk) {
        Write-Host " Next step: run GNN pretraining" -ForegroundColor Yellow
        Write-Host "   .\scripts\train_gnn_pretrain.ps1 --max-samples 500 --cpu-friendly" -ForegroundColor White
    }
    elseif (-not $finetuneOk) {
        Write-Host " Next step: run fine-tuning on Theophylline" -ForegroundColor Yellow
        Write-Host "   src\.venv\Scripts\python src\training\finetune_gnn_pbpk_theoph.py" -ForegroundColor White
    }
}

Write-Host "========================================`n" -ForegroundColor Cyan
