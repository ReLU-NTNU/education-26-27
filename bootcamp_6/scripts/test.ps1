$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run.ps1") test @args
exit $LASTEXITCODE
