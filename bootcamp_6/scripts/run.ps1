# Shared PowerShell launcher; compatible with Windows PowerShell 5.1 and PowerShell 7.
$ErrorActionPreference = "Stop"
$BootcampRoot = Split-Path -Parent $PSScriptRoot
$UvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($UvCommand) {
    $BootcampUv = $UvCommand.Source
} elseif ($env:USERPROFILE -and (Test-Path (Join-Path $env:USERPROFILE ".local\bin\uv.exe"))) {
    $BootcampUv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
} else {
    Write-Host "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    Write-Host "Then rerun this command. You do not need to install Python separately."
    exit 1
}
$PreviousEnvironment = $env:UV_PROJECT_ENVIRONMENT
$PreviousUtf8 = $env:PYTHONUTF8
try {
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $BootcampRoot ".venv"
    $env:PYTHONUTF8 = "1"
    & $BootcampUv run --locked --directory $BootcampRoot --project $BootcampRoot --python cpython-3.13-windows-x86_64-none python (Join-Path $PSScriptRoot "manage.py") @args
    $BootcampExitCode = $LASTEXITCODE
} finally {
    $env:UV_PROJECT_ENVIRONMENT = $PreviousEnvironment
    $env:PYTHONUTF8 = $PreviousUtf8
}
exit $BootcampExitCode
