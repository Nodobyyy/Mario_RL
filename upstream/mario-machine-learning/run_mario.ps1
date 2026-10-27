param(
    [ValidateSet("train", "eval", "professional")]
    [string]$Mode = "train",
    [int]$Timesteps = 20000,
    [string]$RunDir = "",
    [string]$Model = "",
    [int]$PreviewFreq = 5000,
    [int]$PreviewSteps = 512,
    [switch]$LivePreview,
    [string]$ActionSet = "right_only",
    [string]$Seeds = "42"
)

$GpuPython = "C:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe"
$CpuPython = "C:\Users\31660\mario-rl\.venv\Scripts\python.exe"
$Python = if (Test-Path $GpuPython) { $GpuPython } else { $CpuPython }

if ($Mode -eq "train") {
    $args = @(
        "C:\Users\31660\mario-rl\train_mario.py",
        "--timesteps", $Timesteps,
        "--run-dir", $RunDir,
        "--preview-freq", $PreviewFreq,
        "--preview-steps", $PreviewSteps,
        "--action-set", $ActionSet
    )
    if ($LivePreview) {
        $args += "--live-preview"
    }
    & $Python @args
    exit $LASTEXITCODE
}

if ($Mode -eq "professional") {
    $args = @(
        "C:\Users\31660\mario-rl\train_mario_professional.py",
        "--total-timesteps", $Timesteps,
        "--seeds", $Seeds,
        "--preview-freq", $PreviewFreq,
        "--preview-steps", $PreviewSteps
    )
    if ($RunDir) {
        $args += @("--base-dir", $RunDir)
    }
    if ($LivePreview) {
        $args += "--live-preview"
    }
    & $Python @args
    exit $LASTEXITCODE
}

if (-not $Model) {
    Write-Error "When Mode=eval you must pass -Model <path-to-zip>."
    exit 1
}

& $Python "C:\Users\31660\mario-rl\evaluate_mario.py" --model $Model --deterministic
exit $LASTEXITCODE

