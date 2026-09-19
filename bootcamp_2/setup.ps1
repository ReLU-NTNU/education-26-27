$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}
& uv run --locked --project $PSScriptRoot python (Join-Path $PSScriptRoot "lecture.py") setup @args
exit $LASTEXITCODE
