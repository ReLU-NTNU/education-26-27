# Bootcamp 2 · Python for machine learning

An editable **marimo** notebook with 27 exercises covering NumPy, pandas,
Matplotlib and a small PyTorch classifier. No Jupyter or GPU is required.

## Get started

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs the
required Python 3.13 and creates `.venv` automatically. The first setup needs
internet and downloads several hundred MB, mostly CPU PyTorch.

```sh
git clone https://github.com/ReLU-NTNU/education-26-27.git
cd education-26-27/bootcamp_2
```

**macOS / Linux**

```sh
./setup.sh
./start.sh
```

**Windows — PowerShell**

```powershell
.\setup.ps1
.\start.ps1
```

If PowerShell blocks scripts, use these equivalent commands without changing
your execution policy:

```powershell
uv run --locked python lecture.py setup
uv run --locked python lecture.py start
```

The start command opens marimo's editor in your browser. Keep the terminal open;
press Ctrl+C there to stop the server. If the browser doesn't open, use the local
URL printed in the terminal. For a busy port, use `./start.sh --port 2720` or
`.\start.ps1 --port 2720`.

## Work on the exercises

- Edit the `WRITE HERE` sections in `exercises.py` in the marimo editor.
- Replace `None` with your answer; press **Shift+Enter** to run a cell.
- Incomplete answers show a reminder. Checks run once answers are filled in.
  Incorrect answers still produce useful assertion errors.
- Dependent cells update when you change an answer. marimo allows each global
  variable to be defined in one cell; `_name` variables are local to a cell.
- Save with **Ctrl+S**, or **Cmd+S** on macOS. Your work stays in `exercises.py`.
- Click **Load / download MNIST** when you reach the image section. The first
  load downloads MNIST; later loads reuse `data/`.

The notebook intentionally contains unsolved exercises. A successful setup test
means the environment and notebook work, not that the exercises are solved.

## Worked solutions

The complete [solution notebook](solution/solutions.py) lives in `solution/`.
Open it with `./start.sh --solution` or `.\start.ps1 --solution` on Windows.
It includes all exercises and a two-epoch CPU training loop. The data download
and training each have a button, so opening it doesn't start a training run.

## Test before the lecture

```sh
./test.sh
./test.sh --data
./test.sh --solution
```

On Windows use `.\test.ps1` and `.\test.ps1 --data`, or on any OS:

```sh
uv run --locked python lecture.py test
uv run --locked python lecture.py test --data
```

The basic test checks marimo's dependency graph, runs the blank notebook,
renders a plot, checks torchvision's compiled extension, and performs a CPU
training step. `--data` also downloads/verifies MNIST and tests a real image
batch. Tests don't modify the notebook or fill in student answers.

`--solution` runs all 27 worked solutions, including two training epochs on MNIST,
and checks that validation accuracy exceeds 70%. It also includes the data test.
Use `.\test.ps1 --solution` on Windows.

To download the data ahead of class without running the tests:

```sh
uv run --locked python lecture.py data
```

All launchers use `uv run --locked`; an out-of-date lockfile causes an error
instead of silently changing the class environment. No environment activation
or notebook-kernel selection is needed. After downloading dependencies and
MNIST, `uv run --offline --locked python lecture.py start` also works offline.

## Source and maintenance

Adapted from ReLU NTNU's
[2025 Bootcamp 1 exercises](https://github.com/ReLU-NTNU/tutorials-25-26/blob/main/notebooks_2025/bootcamp_1/bootcamp_1_exercises.ipynb)
at commit `2a5a9fdb48549b3a620b7ba3e8bfac6f3d20f5f2`, used here for Bootcamp 2.
The learning sequence and exercise blanks are preserved. Changes cover marimo
cell dependencies, clearer unfinished-answer feedback, explicit data loading,
plot display, and small corrections to inconsistent instructions/checks.

Dependencies live in `pyproject.toml`; exact resolutions are committed in
`uv.lock`. After deliberately changing dependencies, run `uv lock`, then test
again. CPU PyTorch uses its CPU index on Linux/Windows and PyPI on macOS.

Linux has been tested locally. The PowerShell launchers and macOS installation
path should also be tested on those systems before class.
