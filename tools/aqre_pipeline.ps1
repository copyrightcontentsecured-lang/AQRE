param([switch]$Rebuild)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..").Path

function RunPy([string]$path) {
  Write-Host "[RUN] $path" -ForegroundColor Cyan
  & python $path 2>&1 | ForEach-Object { $_.ToString() }
  if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE ($path)" }
}

try {
  if ($Rebuild) { RunPy "$repo\tools\generate_mock_data_v2.py" }

  RunPy "$repo\src\data\build_features.py"
  RunPy "$repo\tools\xg_preflight.py"
  RunPy "$repo\src\models\train_models.py"
  RunPy "$repo\src\models\loo_eval.py"

  $sum = Join-Path $repo "reports\loo_summary.json"
  if (Test-Path $sum) {
    Write-Host "`n[REPORT] loo_summary.json" -ForegroundColor Magenta
    Get-Content -Raw $sum | ConvertFrom-Json | Format-List
  } else {
    Write-Host "[INFO] loo_summary.json bulunamadı." -ForegroundColor Yellow
  }

  Write-Host "`n[DONE] AQRE pipeline tamam." -ForegroundColor Green
} catch {
  Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
  exit 2
}

exit 0
