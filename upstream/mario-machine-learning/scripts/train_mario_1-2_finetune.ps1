# Fine-tune the shipped 1-1 PPO on World 1-2 (same recipe family as P10 time-penalty run).
# Usage: from repo root, .\scripts\train_mario_1-2_finetune.ps1
# Requires: GPU recommended, .venv-gpu activated, `pip install -r requirements.txt`

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$runDir = Join-Path $RepoRoot ("runs/mario_1-2_finetune_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$resume = Join-Path $RepoRoot "pretrained/ppo_mario_1-1/best_eval.zip"
if (-not (Test-Path -LiteralPath $resume)) {
    throw "Missing $resume — clone repo with pretrained weights or set --resume path."
}

python train_mario.py `
  --run-dir $runDir `
  --resume-model $resume `
  --world 1 --stage 2 `
  --timesteps 2000000 `
  --device cuda --action-set simple `
  --n-envs 8 --gamma 0.9 --gae-lambda 0.95 `
  --learning-rate 2.5e-4 --n-steps 512 --batch-size 256 `
  --ent-coef 0.018 --n-epochs 4 --clip-range 0.2 `
  --normalize-reward --norm-reward-clip 10 `
  --forward-reward-scale 0.15 --backward-penalty-scale 0.05 `
  --flag-bonus 200 --stall-steps 120 --stall-penalty 0 `
  --milestone-step 50 --milestone-bonus 0.5 `
  --hurdle-x 400 800 1200 1600 2000 2400 `
  --hurdle-bonus 10 `
  --time-penalty-per-step 0.02 `
  --action-bias-jump 1.0 --rnd-coef 0.3 `
  --eval-freq 50000 --eval-episodes 3 --eval-steps 4000 `
  --preview-freq 100000 `
  --checkpoint-freq 50000

Write-Host "Training launched. Run dir: $runDir"
