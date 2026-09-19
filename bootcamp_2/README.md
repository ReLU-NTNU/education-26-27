# Bootcamp 2 · Python for machine learning

27 exercises in NumPy, pandas, Matplotlib and CPU PyTorch, using **marimo**.

```text
bootcamp_2/
├── solution/        Worked solution notebook
├── scripts/         Setup, launch and test commands
├── tests/           Instructor validation checks
├── work/            Your saved answers (created locally, Git-ignored)
├── pyproject.toml   Dependencies
└── uv.lock          Exact dependency versions
```

The exercise template lives at [`exercises.py`](../exercises.py), in the repository root.

Students use the setup/start scripts and solve `work/exercises.py`.
The Python helper in `scripts/` and checks in `tests/` are maintenance code.

## 1. Set up

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/ReLU-NTNU/education-26-27.git
cd education-26-27/bootcamp_2
```

| | macOS / Linux | Windows PowerShell |
|---|---|---|
| Set up once | `./scripts/setup.sh` | `.\scripts\setup.ps1` |
| Start or resume | `./scripts/start.sh` | `.\scripts\start.ps1` |

uv installs Python 3.13 and the locked dependencies in `.venv`. The first setup
needs internet and downloads several hundred MB. No GPU, Jupyter, environment
activation, or kernel selection is needed.

If PowerShell blocks scripts, run these equivalent commands:

```sh
uv run --locked python scripts/manage.py setup
uv run --locked python scripts/manage.py start
```

## 2. Solve and save

The launcher creates **`work/exercises.py`** the first time and opens it in
marimo's browser editor. Every later launch reopens that same file.

1. Fill in the `WRITE HERE` blanks. Replace `None` with your answer.
2. Press **Shift+Enter** to run the cell and its checks.
3. Save with **Ctrl+S** (macOS: **Cmd+S**).
4. Run the start command again whenever you want to resume.

Your answers are in `work/exercises.py`, not the tracked [`../exercises.py`](../exercises.py) template at the repository root.
Setup, restart, and `git pull` do not overwrite that working copy. `work/` is
Git-ignored: back it up separately if you want to keep your answers elsewhere.

marimo updates dependent cells when values change. Define each global variable
in only one cell; `_name` variables are local to a cell. Unfinished answers show
a reminder, and incorrect answers produce assertion errors.

Keep the terminal open while working. Stop the server with **Ctrl+C**. If a
browser doesn't open, follow the local URL in the terminal. For another port,
use `./scripts/start.sh --port 2720` or `.\scripts\start.ps1 --port 2720`.

**VS Code:** install the official [marimo extension](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo),
open `work/exercises.py` as a marimo notebook, and select this bootcamp's `.venv`.

## 3. Load data when needed

Click **Load / download MNIST** when you reach the image section. It downloads
once and reuses the cache. Data never needs to be committed or copied into this
repo. The exercise and solution notebooks share the same cache.

Typical cache locations:

- Linux: `~/.cache/relu-education/datasets/`
- macOS: `~/Library/Caches/relu-education/datasets/`
- Windows: `%LOCALAPPDATA%\relu-education\Cache\datasets\`

`./scripts/setup.sh` (or `.\scripts\setup.ps1`) prints the exact path on your machine.
To download ahead of class:

```sh
uv run --locked python scripts/manage.py data
```

After dependencies and data are cached, start offline with:

```sh
uv run --offline --locked python scripts/manage.py start
```

## Check your setup

Run `./scripts/test.sh` on macOS/Linux or `.\scripts\test.ps1` on Windows. This checks the notebook,
plotting, CPU training, repository hygiene, and preservation of saved work.
Add `--data` to also check the MNIST download and a real image batch.

## Solutions and instructor checks

Try the exercises before opening the [worked solutions](solution/README.md).
Use `./scripts/start.sh --solution` (Windows: `.\scripts\start.ps1 --solution`) to open a personal
copy in `work/solutions.py`. The tracked solution stays unchanged.

`./scripts/test.sh --solution` runs all 27 solutions and two training epochs.
Windows equivalent: `.\scripts\test.ps1 --solution`.

## Maintenance

The source is ReLU NTNU's [2025 Bootcamp 1 exercises](https://github.com/ReLU-NTNU/tutorials-25-26/blob/main/notebooks_2025/bootcamp_1/bootcamp_1_exercises.ipynb)
at commit `2a5a9fdb48549b3a620b7ba3e8bfac6f3d20f5f2`, adapted for Bootcamp 2.

Commit teaching source, scripts, docs and `uv.lock`. Do not commit datasets,
student answers, environments, outputs or model checkpoints. Ignore rules and
`../scripts/check_repository.py` cover these generated artifacts.

After changing dependencies in `pyproject.toml`, run `uv lock`, then test again.
Launchers use `--locked` and will reject an out-of-date lockfile. Setup and
exercise execution are tested on Linux; Windows/macOS still need native testing.
