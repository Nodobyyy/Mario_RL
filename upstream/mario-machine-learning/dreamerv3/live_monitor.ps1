# Live monitor: refresh every 20s
Set-Location "c:\Users\31660\mario-rl\dreamerv3"
$python = "c:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe"
while ($true) {
  Clear-Host
  Write-Host "=== MARIO LIVE MONITOR === $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
  Write-Host ""
  .\status.ps1
  Write-Host ""
  & $python tracker.py
  Write-Host ""
  Write-Host "(auto refresh every 20s, Ctrl+C to stop)" -ForegroundColor Yellow
  Start-Sleep -Seconds 20
}
