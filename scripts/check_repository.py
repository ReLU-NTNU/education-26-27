"""Reject accidentally tracked datasets, environments and generated artifacts."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRECTORIES = {
    "data", "datasets", "MNIST", ".venv", "venv", "__pycache__", "__marimo__",
    "work", "outputs", "checkpoints", ".cache", ".test-output",
}
DATA_EXTENSIONS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".csv", ".parquet", ".npy", ".npz",
    ".h5", ".hdf5", ".zip", ".gz", ".tar",
}


def main() -> int:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    blocked = []
    for name in filter(None, tracked):
        path = Path(name)
        if GENERATED_DIRECTORIES.intersection(path.parts) or path.suffix.lower() in DATA_EXTENSIONS:
            blocked.append(name)
        elif (ROOT / path).is_file() and (ROOT / path).stat().st_size > 5_000_000:
            blocked.append(name)
    if blocked:
        print("Do not commit datasets, student work, environments or generated artifacts:")
        print("\n".join(blocked))
        return 1
    print("PASS: Git tracks only teaching source, documentation and setup files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
