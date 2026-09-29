$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run.ps1") setup @args
exit $LASTEXITCODE
