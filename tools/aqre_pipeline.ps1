param([switch]$Rebuild)

$ErrorActionPreference = "Stop"
try { chcp 65001 > $null } catch {}
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"

# Repo kökü ve PYTHONPATH (src importları için)
$root = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = $root

function RunMod([string]$mod) {
  Write-Host "[RUN] python -m $mod" -ForegroundColor Cyan
  & python -X utf8 -u -m $mod
  if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE ($mod)" }
}

try {
  if ($Rebuild) { RunMod 'tools.generate_mock_data_v2' }

  RunMod 'src.data.build_features'
  RunMod 'tools.xg_preflight'
  RunMod 'src.models.train_models'
  RunMod 'src.models.loo_eval'

  Write-Host "`n[DONE] AQRE pipeline tamam." -ForegroundColor Green
} catch {
  Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
  exit 2
}
