# diagnose_latest.ps1
# -------------------
# Roll out the latest periodic PPO checkpoint with both stochastic and
# deterministic policies, write best.mp4 + all_runs.json under
# <run>/diagnose_<step>/ . Use this to peek at "what is the current
# policy actually doing" while training is still in flight.
#
# Usage:
#   pwsh -ExecutionPolicy Bypass -File .\diagnose_latest.ps1
#   pwsh -ExecutionPolicy Bypass -File .\diagnose_latest.ps1 -Seeds 5 -MaxSteps 1200

param(
    [int]$Seeds = 3,
    [int]$MaxSteps = 800,
    [string]$RunDir = "",
    [string]$Modes = "stochastic,deterministic"
)

$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\31660\mario-rl"
$python = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
$renderer = Join-Path $projectRoot "render_best_checkpoint.py"

function Get-LatestPpoRunDir {
    $candidates = Get-ChildItem (Join-Path $projectRoot "runs") -Directory -Filter "ppo_*" `
        -ErrorAction SilentlyContinue
    if (-not $candidates) { return $null }
    $scored = foreach ($d in $candidates) {
        $m = [regex]::Match($d.Name, "(\d{8}_\d{6})$")
        if ($m.Success) {
            [PSCustomObject]@{ Dir = $d; Stamp = $m.Groups[1].Value }
        }
    }
    ($scored | Sort-Object Stamp -Descending | Select-Object -First 1).Dir
}

if (-not $RunDir) {
    $runDirObj = Get-LatestPpoRunDir
    if (-not $runDirObj) { throw "no runs/ppo_* directory found" }
    $RunDir = $runDirObj.FullName
}

$ckptDir = Join-Path $RunDir "run\models\checkpoints"
if (-not (Test-Path $ckptDir)) { throw "no checkpoints under $ckptDir" }
$latest = Get-ChildItem $ckptDir -Filter "mario_ppo_*_steps.zip" |
    Sort-Object { [int](($_.BaseName -split "_")[2]) } -Descending |
    Select-Object -First 1
if (-not $latest) { throw "no mario_ppo_*_steps.zip in $ckptDir" }

$outDir = Join-Path $RunDir "run\diagnose_$($latest.BaseName)"
Write-Host "checkpoint : $($latest.FullName)"
Write-Host "output_dir : $outDir"
Write-Host "seeds      : $Seeds  max_steps: $MaxSteps  modes: $Modes"
Write-Host "---"

& $python $renderer `
    --model $latest.FullName `
    --seeds $Seeds `
    --max-steps $MaxSteps `
    --policy-modes $Modes `
    --output-dir $outDir
