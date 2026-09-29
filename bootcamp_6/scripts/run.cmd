@echo off
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "BOOTCAMP_ROOT=%%~fI"
set "BOOTCAMP_UV=uv.exe"
where uv.exe >nul 2>nul
if not errorlevel 1 goto found_uv
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "BOOTCAMP_UV=%USERPROFILE%\.local\bin\uv.exe"
    goto found_uv
)
echo Install uv first: https://docs.astral.sh/uv/getting-started/installation/ 1>&2
echo Then rerun this command. You do not need to install Python separately. 1>&2
exit /b 1
:found_uv
set "UV_PROJECT_ENVIRONMENT=%BOOTCAMP_ROOT%\.venv"
set "PYTHONUTF8=1"
"%BOOTCAMP_UV%" run --locked --directory "%BOOTCAMP_ROOT%" --project "%BOOTCAMP_ROOT%" --python cpython-3.13-windows-x86_64-none python "%BOOTCAMP_ROOT%\scripts\manage.py" %*
exit /b %ERRORLEVEL%
