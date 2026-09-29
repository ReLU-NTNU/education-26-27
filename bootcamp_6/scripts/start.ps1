$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run.ps1") start @args
exit $LASTEXITCODE
