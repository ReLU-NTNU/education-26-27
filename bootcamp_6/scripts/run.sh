#!/usr/bin/env sh
# Shared POSIX launcher: macOS (Intel/Apple Silicon) and Linux.
set -eu
BOOTCAMP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if command -v uv >/dev/null 2>&1; then
    BOOTCAMP_UV=$(command -v uv)
elif [ -x "${HOME:-}/.local/bin/uv" ]; then
    BOOTCAMP_UV="$HOME/.local/bin/uv"
else
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    echo "Then rerun this command. You do not need to install Python separately." >&2
    exit 1
fi
# Keep this workbook isolated even if another environment is active.
export UV_PROJECT_ENVIRONMENT="$BOOTCAMP_ROOT/.venv"
export PYTHONUTF8=1
exec "$BOOTCAMP_UV" run --locked --directory "$BOOTCAMP_ROOT" --project "$BOOTCAMP_ROOT" --python 3.13 python "$BOOTCAMP_ROOT/scripts/manage.py" "$@"
