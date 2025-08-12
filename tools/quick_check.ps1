# tools/quick_check.ps1
$ErrorActionPreference = "Stop"

# 1) 安裝相依（無互動、安靜）
if (Test-Path "requirements.txt") {
  $env:PIP_NO_INPUT = "1"
  pip install -r requirements.txt -q
}

# 2) Lint（有裝就跑，沒裝就跳過）
try { ruff --version > $null 2>&1; ruff check . --quiet } catch {}

# 3) 超快速測試（只跑 smoke；沒測試就略過）
if (Test-Path "tests") {
  try { pytest -q -m "smoke" --maxfail=1 --disable-warnings } catch { exit 1 }
}

Write-Host "✅ Quick check done."
