# Launch high-value run: DreamerV3 + Plan2Explore, warm-started from rescue checkpoint.
$ErrorActionPreference = "Continue"
Set-Location "c:\Users\31660\mario-rl\dreamerv3"

$src = "logs\mario_run3_rescue"
$dst = "logs\mario_run4_plan2explore"

if (-not (Test-Path "$src\latest.pt")) {
  Write-Host "ERROR: $src\latest.pt not found" -ForegroundColor Red
  exit 1
}

New-Item -ItemType Directory -Force -Path "$dst\train_eps" | Out-Null
New-Item -ItemType Directory -Force -Path "$dst\eval_eps"  | Out-Null

Copy-Item "$src\latest.pt" "$dst\latest.pt" -Force

# copy recent episodes as warm replay context
$recentTrain = Get-ChildItem "$src\train_eps\*.npz" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 40
foreach ($f in $recentTrain) { Copy-Item $f.FullName "$dst\train_eps\" -Force }
$recentEval = Get-ChildItem "$src\eval_eps\*.npz" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 10
foreach ($f in $recentEval) { Copy-Item $f.FullName "$dst\eval_eps\" -Force }

Write-Host "Starting mario_run4_plan2explore..." -ForegroundColor Cyan
$env:PYTHONUNBUFFERED = "1"
Start-Process -FilePath "c:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe" -ArgumentList @("dreamer.py","--configs","mario_plan2explore","--logdir","$dst") -RedirectStandardOutput "logs\run4_p2e.log" -RedirectStandardError "logs\run4_p2e.err" -NoNewWindow -PassThru | Select-Object Id, ProcessName
