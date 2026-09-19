"""Cross-platform commands; run with uv run --locked python scripts/manage.py COMMAND."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def working_copy(solution: bool = False, root: Path = ROOT) -> Path:
    """Create a personal notebook once; never overwrite saved student work."""
    source = root / "solution" / "solutions.py" if solution else root / "notebooks" / "exercises.py"
    target = root / "work" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_text(encoding="utf-8")
    try:
        with target.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError:
        pass
    return target


def data_directory() -> Path:
    from platformdirs import user_cache_path

    return user_cache_path("relu-education", appauthor=False) / "datasets"


def setup() -> None:
    for package in ("marimo", "numpy", "pandas", "matplotlib", "torch", "torchvision"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Your exercise notebook: {working_copy()}")
    print(f"Dataset cache (outside the repository): {data_directory()}")
    print("Ready. Run ./scripts/start.sh (macOS/Linux) or .\\scripts\\start.ps1 (Windows).")


def download_data():
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train = datasets.MNIST(data_directory(), train=True, download=True, transform=transform)
    test = datasets.MNIST(data_directory(), train=False, download=True, transform=transform)
    assert len(train) == 60000 and len(test) == 10000
    image, label = train[0]
    assert image.shape == (1, 28, 28) and 0 <= label <= 9
    print(f"MNIST ready: {len(train):,} train / {len(test):,} test images.")
    return train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "start", "test", "data"))
    parser.add_argument("--data", action="store_true", help="also test/download MNIST")
    parser.add_argument("--solution", action="store_true", help="open or fully test the worked solutions")
    args, extra = parser.parse_known_args()
    if extra and args.command != "start":
        parser.error(f"Unrecognized arguments: {' '.join(extra)}")
    os.chdir(ROOT)
    if args.command == "setup":
        setup()
        if args.data:
            download_data()
    elif args.command == "data":
        download_data()
    elif args.command == "test":
        flags = (["--data"] if args.data else []) + (["--solution"] if args.solution else [])
        return subprocess.call([sys.executable, str(ROOT / "tests" / "check_setup.py"), *flags])
    else:
        notebook_path = working_copy(args.solution)
        print(f"Opening your saved working copy: {notebook_path}", flush=True)
        return subprocess.call([
            sys.executable, "-m", "marimo", "edit", str(notebook_path),
            "--host", "127.0.0.1", "--skip-update-check", *extra,
        ])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
