"""Create a personal workbook once and launch it without overwriting answers."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def working_copy(solution: bool = False, root: Path = ROOT) -> Path:
    folder = "solutions" if solution else "exercises"
    target = root / "work" / ("solutions.py" if solution else "exercises.py")
    content = (root / folder / "workbook.py").read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except FileExistsError:
        pass
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "start", "test"])
    parser.add_argument("--solution", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and args.command != "start":
        parser.error(f"Unknown arguments: {' '.join(extra)}")
    os.chdir(ROOT)
    if args.command == "setup":
        for package in ["marimo", "numpy", "pandas", "matplotlib", "scikit-learn"]:
            print(f"{package}: {importlib.metadata.version(package)}")
        print(f"Your saved notebook: {working_copy(args.solution)}")
        print("Ready. Run ./scripts/start.sh or .\\scripts\\start.ps1.")
        return 0
    if args.command == "test":
        return subprocess.call([sys.executable, str(ROOT / "tests" / "check_setup.py")])
    target = working_copy(args.solution)
    print(f"Opening your saved notebook: {target}", flush=True)
    return subprocess.call([sys.executable, "-m", "marimo", "edit", str(target),
                            "--host", "127.0.0.1", "--skip-update-check", *extra])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
