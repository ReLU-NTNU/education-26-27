"""Reuse the existing environment and create-once workbook launcher."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--setup", action="store_true", help="Prepare a copy for VS Code without opening a browser.")
    args, extra = parser.parse_known_args()
    if args.setup and extra:
        parser.error("Extra marimo arguments are only supported when opening the browser.")
    shared_path = ROOT.parent / "bootcamp_6/scripts/manage.py"
    spec = importlib.util.spec_from_file_location("bootcamp_workbook_manager", shared_path)
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    target = manager.working_copy(solution=args.solution, root=ROOT)
    print(f"Your saved workbook: {target}", flush=True)
    if args.setup:
        return 0
    return subprocess.call([
        sys.executable, "-m", "marimo", "edit", str(target),
        "--host", "127.0.0.1", "--skip-update-check", *extra,
    ])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
