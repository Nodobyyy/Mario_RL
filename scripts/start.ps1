$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv311/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & (Join-Path $PSScriptRoot 'setup.ps1')
}
& $pythonExe (Join-Path $PSScriptRoot 'serve.py')
if ($LASTEXITCODE -ne 0) { Read-Host 'Server exited. Press Enter to close' }
