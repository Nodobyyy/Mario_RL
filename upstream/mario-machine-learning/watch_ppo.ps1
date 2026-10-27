# =====================================================================
# watch_ppo.ps1 -- one-shot health snapshot for the active PPO trainer
# =====================================================================
# Usage:   .\watch_ppo.ps1                           # auto-detect newest run
#          .\watch_ppo.ps1 -RunDir <path>            # specific run dir
#          .\watch_ppo.ps1 -Loop -EveryS 30          # poll every 30s
# =====================================================================
param(
  [string] $RunDir = "",
  [switch] $Loop,
  [int]    $EveryS = 20
)

function Get-LatestRunDir {
  # Select the newest run directory by the timestamp encoded in its name
  # (NOT by file mtime, because old runs can have late writes during shutdown).
  $candidates = @()
  $candidates += Get-ChildItem 'C:\Users\31660\mario-rl\runs' -Directory -ErrorAction SilentlyContinue `
    | Where-Object { $_.Name -match '^ppo_(safe|overnight|professional|.+)_(\d{8}_\d{6})' }
  if (-not $candidates) { return $null }
  $best = $candidates | Sort-Object {
    if ($_.Name -match '_(\d{8}_\d{6})$') { $matches[1] } else { '0' }
  } -Descending | Select-Object -First 1
  $best.FullName
}

function Show-Snapshot {
  param([string] $Dir)

  $stdout = Join-Path $Dir 'train.out.log'
  $stderr = Join-Path $Dir 'train.err.log'

  Write-Host ""
  Write-Host ("=" * 78)
  Write-Host (" PPO health snapshot @ {0}" -f (Get-Date -Format 'HH:mm:ss'))
  Write-Host ("   run dir : {0}" -f $Dir)
  Write-Host ("=" * 78)

  if (Test-Path $stdout) {
    Write-Host "--- last training block ---"
    Get-Content $stdout -Tail 30 | ForEach-Object { Write-Host $_ }
  } else {
    Write-Host "stdout log not found yet."
  }

  $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" `
    | Where-Object { $_.CommandLine -like '*train_mario*' }
  Write-Host ""
  Write-Host "--- workers ---"
  if ($procs) {
    $procs | Select-Object ProcessId, ParentProcessId,
      @{N='Cmd';E={ if($_.CommandLine.Length -gt 70){$_.CommandLine.Substring(0,70)+'...'}else{$_.CommandLine} }} `
      | Format-Table -AutoSize | Out-String | Write-Host
  } else {
    Write-Host "  (no train_mario python process running)"
  }

  Write-Host "--- gpu ---"
  & 'C:\Windows\System32\nvidia-smi.exe' --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null | ForEach-Object {
    Write-Host "  $_"
  }

  $evalDir = Join-Path $Dir 'run\evaluations'
  if (Test-Path $evalDir) {
    $bestSummary = Join-Path $evalDir 'best_eval_summary.json'
    if (Test-Path $bestSummary) {
      Write-Host "--- best_eval_summary ---"
      Get-Content $bestSummary -Tail 20 | ForEach-Object { Write-Host "  $_" }
    }
  }
}

if (-not $RunDir) { $RunDir = Get-LatestRunDir }
if (-not $RunDir) {
  Write-Host "No PPO run directory found under C:\Users\31660\mario-rl\runs\"
  exit 1
}

if ($Loop) {
  while ($true) {
    Show-Snapshot -Dir $RunDir
    Write-Host "(sleeping ${EveryS}s, Ctrl+C to stop)"
    Start-Sleep -Seconds $EveryS
  }
} else {
  Show-Snapshot -Dir $RunDir
}
