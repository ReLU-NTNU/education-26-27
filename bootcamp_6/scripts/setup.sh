#!/usr/bin/env sh
set -eu
BOOTCAMP_SCRIPTS=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$BOOTCAMP_SCRIPTS/run.sh" setup "$@"
