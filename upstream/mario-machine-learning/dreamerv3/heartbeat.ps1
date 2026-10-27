param(
    [int]$IntervalMinutes = 10,
    [switch]$Once,
    [string]$RunLogPath = "c:\Users\31660\mario-rl\dreamerv3\logs\run4_p2e.log",
    [string]$MetricsPath = "c:\Users\31660\mario-rl\dreamerv3\logs\mario_run4_plan2explore\metrics.jsonl",
    [string]$OutPath = "c:\Users\31660\mario-rl\dreamerv3\logs\heartbeat.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

function Get-LatestStep([string]$logPath) {
    if (-not (Test-Path $logPath)) { return $null }
    $last = Select-String -Path $logPath -Pattern '^\[(\d+)\]' | Select-Object -Last 1
    if ($last -and $last.Line -match '^\[(\d+)\]') {
        return [int]$Matches[1]
    }
    return $null
}

function Get-LatestEval([string]$logPath) {
    if (-not (Test-Path $logPath)) { return $null }
    $last = Select-String -Path $logPath -Pattern 'eval_return' | Select-Object -Last 1
    if ($last -and $last.Line -match '^\[(\d+)\].*eval_return\s+(-?\d+\.?\d*)') {
        return [pscustomobject]@{
            Step   = [int]$Matches[1]
            Return = [double]$Matches[2]
            Line   = $last.Line
        }
    }
    return $null
}

function Get-TrainingProc() {
    $procs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WS -gt 500MB }
    if (-not $procs) { return $null }
    return $procs | Sort-Object WS -Descending | Select-Object -First 1
}

function Write-Heartbeat() {
    $now = Get-Date
    $proc = Get-TrainingProc
    $step = Get-LatestStep -logPath $RunLogPath
    $eval = Get-LatestEval -logPath $RunLogPath
    $metricsAgeSec = $null
    if (Test-Path $MetricsPath) {
        $metricsAgeSec = [int]((Get-Date) - (Get-Item $MetricsPath).LastWriteTime).TotalSeconds
    }

    $status = if ($proc) { "alive" } else { "down" }
    $procId = if ($proc) { $proc.Id } else { $null }
    $cpuSec = if ($proc) { [math]::Round($proc.CPU, 1) } else { $null }
    $memMB = if ($proc) { [math]::Round($proc.WS / 1MB, 1) } else { $null }
    $warn = @()
    if (-not $proc) { $warn += "python_not_running" }
    if ($null -ne $metricsAgeSec -and $metricsAgeSec -gt 900) { $warn += "metrics_stale" }
    if ($null -eq $step) { $warn += "no_step_found" }

    $record = [pscustomobject]@{
        timestamp      = $now.ToString("yyyy-MM-dd HH:mm:ss")
        status         = $status
        pid            = $procId
        cpu_sec        = $cpuSec
        mem_mb         = $memMB
        latest_step    = $step
        latest_eval    = if ($eval) { $eval.Return } else { $null }
        latest_eval_at = if ($eval) { $eval.Step } else { $null }
        metrics_age_s  = $metricsAgeSec
        warn           = ($warn -join ",")
    }

    $line = ($record | ConvertTo-Json -Compress)
    Add-Content -Path $OutPath -Value $line

    $color = if ($warn.Count -eq 0) { "Green" } else { "Yellow" }
    Write-Host ("[{0}] status={1} step={2} eval={3}@{4} pid={5} mem={6}MB warn={7}" -f `
        $record.timestamp, $record.status, $record.latest_step, $record.latest_eval, `
        $record.latest_eval_at, $record.pid, $record.mem_mb, `
        ($(if ($record.warn) { $record.warn } else { "-" }))) -ForegroundColor $color
}

if ($Once) {
    Write-Heartbeat
    exit 0
}

Write-Host ("Heartbeat started. Interval={0} min. Output={1}" -f $IntervalMinutes, $OutPath) -ForegroundColor Cyan
while ($true) {
    Write-Heartbeat
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
