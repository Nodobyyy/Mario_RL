# Copy key 1-1 PPO checkpoints out of runs/ into a dated folder under backups/.
# Usage (from repo root):
#   .\scripts\backup_1-1_checkpoints.ps1
# Optional custom root (e.g. external drive):
#   .\scripts\backup_1-1_checkpoints.ps1 -DestinationRoot "D:\MarioBackups"

param(
    [string]$DestinationRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $RepoRoot "train_mario.py"))) {
    throw "Run from mario-rl repo (train_mario.py not found under $RepoRoot)"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ($DestinationRoot) {
    $dest = Join-Path $DestinationRoot "mario_1-1_checkpoints_$stamp"
} else {
    $dest = Join-Path $RepoRoot "backups\mario_1-1_checkpoints_$stamp"
}

New-Item -ItemType Directory -Path $dest -Force | Out-Null

$sources = @(
    @{
        Label = "P10_timepen"
        RunDir = Join-Path $RepoRoot "runs\ppo_efficient_timepen_20260426_020450"
    },
    @{
        Label = "P9_resume2m"
        RunDir = Join-Path $RepoRoot "runs\ppo_resume2m_ent002_20260425_172812"
    }
)

foreach ($s in $sources) {
    if (-not (Test-Path $s.RunDir)) { continue }
    $sub = Join-Path $dest $s.Label
    New-Item -ItemType Directory -Path $sub -Force | Out-Null
    $files = @(
        (Join-Path $s.RunDir "train_config.json"),
        (Join-Path $s.RunDir "models\mario_final.zip"),
        (Join-Path $s.RunDir "evaluations\best_eval.zip"),
        (Join-Path $s.RunDir "models\vecnormalize.pkl"),
        (Join-Path $s.RunDir "evaluations\vecnormalize.pkl")
    )
    foreach ($f in $files) {
        if (Test-Path -LiteralPath $f) {
            Copy-Item -LiteralPath $f -Destination $sub -Force
        }
    }
}

Write-Host "Backup done: $dest"
Get-ChildItem $dest -Recurse -File | ForEach-Object { $_.FullName }
