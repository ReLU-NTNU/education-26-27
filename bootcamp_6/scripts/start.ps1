$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}
& uv run --locked --project $Root python (Join-Path $PSScriptRoot "manage.py") start @args
exit $LASTEXITCODE
