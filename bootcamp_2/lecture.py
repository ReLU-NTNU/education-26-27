"""Cross-platform commands; run with uv run --locked python lecture.py COMMAND."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def working_copy(solution: bool = False, root: Path = ROOT) -> Path:
    """Create a personal notebook once; never overwrite saved student work."""
    source = root / "solution" / "solutions.py" if solution else root / "exercises.py"
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
    print("Ready. Run ./start.sh (macOS/Linux) or .\\start.ps1 (Windows).")


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


def test(with_data: bool = False, with_solution: bool = False) -> None:
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torchvision import transforms

    # Protect student answers from setup/restart and upstream template changes.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "exercises.py").write_text("template", encoding="utf-8")
        personal = working_copy(root=root)
        assert personal.read_text(encoding="utf-8") == "template"
        personal.write_text("my saved answers", encoding="utf-8")
        (root / "exercises.py").write_text("updated template", encoding="utf-8")
        assert working_copy(root=root).read_text(encoding="utf-8") == "my saved answers"

    subprocess.run([sys.executable, str(ROOT.parent / "scripts" / "check_repository.py")], check=True)
    subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict",
         str(ROOT / "exercises.py"), str(ROOT / "solution" / "solutions.py")],
        check=True,
    )
    # Exercise the libraries together, including torchvision's compiled extension.
    from torchvision.ops import nms

    assert nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([0.9]), 0.5).tolist() == [0]
    assert pd.Series(np.arange(3)).sum() == 3
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    plt.close(fig)
    image = transforms.ToTensor()(np.zeros((28, 28), dtype=np.uint8))
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()(model(image.unsqueeze(0)), torch.tensor([0]))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).item()

    # Run the actual blank student app: unfinished answers must pause cleanly.
    spec = importlib.util.spec_from_file_location("bootcamp_2_exercises", ROOT / "exercises.py")
    assert spec and spec.loader
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    outputs, definitions = notebook.app.run()
    assert outputs and definitions["device"].type == "cpu"
    assert not definitions["load_mnist"].value, "Data loading should be opt-in"
    plt.close("all")

    if with_data:
        train = download_data()
        from types import SimpleNamespace

        # Exercise the notebook's own load-button path, including its data path.
        _, loaded = notebook.app.run(defs={"load_mnist": SimpleNamespace(value=True)})
        assert len(loaded["mnist_train"]) == 60000
        assert len(loaded["mnist_test"]) == 10000
        plt.close("all")
        batch, labels = next(iter(torch.utils.data.DataLoader(train, batch_size=64)))
        assert batch.shape == (64, 1, 28, 28) and labels.dtype == torch.int64
        optimizer.zero_grad()
        data_loss = nn.CrossEntropyLoss()(model(batch), labels)
        data_loss.backward()
        optimizer.step()
        assert torch.isfinite(data_loss).item()
    print("PASS: notebook graph, blank app, plotting, torchvision and CPU training step.")
    if with_data:
        print("PASS: MNIST download/cache, data loader and training step with real images.")

    if with_solution:
        import contextlib
        import io
        import re
        from types import SimpleNamespace

        solution_spec = importlib.util.spec_from_file_location(
            "bootcamp_2_solutions", ROOT / "solution" / "solutions.py"
        )
        assert solution_spec and solution_spec.loader
        solution = importlib.util.module_from_spec(solution_spec)
        solution_spec.loader.exec_module(solution)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, solved = solution.app.run(defs={
                "load_mnist": SimpleNamespace(value=True),
                "train_now": SimpleNamespace(value=True),
            })
        checks = re.findall(r"^OK \d+\.\d+", log.getvalue(), re.MULTILINE)
        assert len(checks) == 27, f"Expected 27 completed exercises, got {len(checks)}"
        assert solved["final_val_acc"] >= 0.70, "Training did not learn above the smoke-test threshold"
        plt.close("all")
        print(f"PASS: all 27 solutions; 2 epochs; validation accuracy {solved['final_val_acc']:.1%}.")


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
        test(args.data or args.solution, args.solution)
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
