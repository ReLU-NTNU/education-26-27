# Bootcamp 6 · KNN and single decision trees

A beginner workbook for the lecture on KNN, decision trees and computational
complexity. About **40 minutes**, plus two optional extensions. Uses the small
Iris dataset bundled with scikit-learn; no dataset download or GPU is needed.

- **[Student workbook](exercises/workbook.py)** — explanations, six fill-in exercises,
  feedback, tree diagrams and confusion matrices.
- **[Worked solutions](solutions/workbook.py)** — completed code and instructor notes.

## Start today's workbook

Already cloned this repository? From its root:

```sh
git pull
cd bootcamp_6
```

First-time users can run:

```sh
git clone https://github.com/ReLU-NTNU/education-26-27.git
cd education-26-27/bootcamp_6
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed.
Then, from `bootcamp_6/`:

| Action | macOS / Linux | Windows PowerShell |
|---|---|---|
| Set up once | `./scripts/setup.sh` | `.\scripts\setup.ps1` |
| Open or resume workbook | `./scripts/start.sh` | `.\scripts\start.ps1` |
| Open worked solutions | `./scripts/start.sh --solution` | `.\scripts\start.ps1 --solution` |
| Run checks | `./scripts/test.sh` | `.\scripts\test.ps1` |

If PowerShell blocks scripts, use `uv run --locked python scripts/manage.py setup`
and `uv run --locked python scripts/manage.py start` instead.

The first setup installs Python 3.13 and locked dependencies in this folder's
`.venv`. To install or synchronize it directly, run `uv sync --locked` from
`bootcamp_6/`. It needs internet for installation; afterward the workbook can run offline.
The launcher opens marimo in your browser. Keep the terminal open; press Ctrl+C
to stop the server. Use `--port 2720` with the start command if needed.

## Work and save

The launcher creates `work/exercises.py` once and reopens it on later launches.
Replace `None` at the `WRITE HERE` comments, run the cell with Shift+Enter, and
save with Ctrl+S / Cmd+S. Unfinished exercises show a reminder instead of an error.
Cells depending on an unfinished answer wait until it is completed.

Solutions open separately as `work/solutions.py`. Setup and start never overwrite
an existing working copy, including after a template update. Personal files under
`work/` are Git-ignored: back them up separately. To adopt a newer template, first
rename your personal notebook to a backup, then run start again.

## What students do

1. Predict a class from a three-neighbour vote.
2. Fit a scaled KNN pipeline and predict validation labels.
3. Compare four values of k using training and validation accuracy.
4. Fit a shallow tree and trace a prediction through its diagram.
5. Compare tree depths and look for overfitting.
6. Compare the selected models on a held-out test set and interpret errors.

Optional: explore the effect of changing feature units and count the work in
brute-force KNN distance computation. The workbook uses a fixed stratified
train/validation/test split and fits scaling only on the appropriate training data.

## Repository layout and compatibility

```text
bootcamp_6/
├── exercises/workbook.py
├── solutions/workbook.py
├── scripts/                 Setup, launch and test commands
├── tests/check_setup.py
├── pyproject.toml
├── uv.lock
├── .venv/                   Local environment, ignored
└── work/                    Personal notebooks, ignored
```

Bootcamp 2 retains its existing files, environment, lockfile and commands.
This workbook is new material for the current lecture.

Maintainers: run `./scripts/test.sh`. It checks the blank notebook, runs the full
solution, checks train-only scaling and model results, and verifies that launching
again preserves saved answers. Tested on macOS; PowerShell launchers follow the
existing Bootcamp 2 pattern but have not been run on Windows.
