param([switch]$Cuda, [switch]$Dev)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.python311/tools/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    New-Item -ItemType Directory -Path (Join-Path $projectRoot 'tmp') -Force | Out-Null
    Write-Host 'Downloading project-local Python 3.11.9 from the official NuGet Python package...'
    $archive = Join-Path $projectRoot 'tmp/python311.zip'
    Invoke-WebRequest -Uri 'https://www.nuget.org/api/v2/package/python/3.11.9' -OutFile $archive -UseBasicParsing
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $projectRoot '.python311') -Force
}
$setupArgs = @((Join-Path $PSScriptRoot 'setup.py'))
if ($Cuda) { $setupArgs += '--cuda' }
if ($Dev) { $setupArgs += '--dev' }
& $pythonExe @setupArgs
if ($LASTEXITCODE -ne 0) { throw 'Setup failed. See the error above.' }
