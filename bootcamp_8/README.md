# Bootcamp 8 · Hyperparameter optimization

The previous random-forest demonstration is now a marimo exercise workbook,
with a separate worked solutions version. It keeps the same flow:

1. Fit and cross-validate a baseline forest on handwritten digits.
2. Tune the forest with grid search.
3. Compare a random search.
4. Build and cross-validate a preprocessing pipeline.

Each section has `WRITE HERE` blanks and discussion questions. Choose the tuned
model using CV scores before the final test comparison. The grid has 135 fold
fits, so allow a few minutes for it; a smaller grid is suggested in the prompt
if needed. The data is included with scikit-learn; no GPU is needed.

## Open the exercises

Get the repository and install **uv** using the [course instructions](../README.md).
From the repository folder, run this command on macOS, Linux or Windows:

```sh
uv run --locked --project bootcamp_6 python bootcamp_8/start.py
```

This reuses the existing Bootcamp 6 environment and opens a personal copy at
`bootcamp_8/work/exercises.py` in your browser. Keep the terminal open while
working; press **Ctrl+C** to stop it. If the browser does not open, follow the
local URL printed in the terminal.

Replace `None` beside each **WRITE HERE** comment and press **Shift+Enter**.
Hints for unfinished exercises are expected; dependent cells wait. Save with
**Ctrl+S / Cmd+S**. Keep your discussion answers in your own notes.

## Use VS Code instead

First prepare the personal copy without opening a browser:

```sh
uv run --locked --project bootcamp_6 python bootcamp_8/start.py --setup
```

Open the repo in VS Code and open `bootcamp_8/work/exercises.py`. Use **marimo:
Open as marimo notebook**, and select the Python kernel in `bootcamp_6/.venv`.
The existing environment is named **relu-bootcamp-six (Python 3.13.x)**. Install
the Python and marimo extensions if needed. If the kernel is missing, select
`bootcamp_6/.venv/bin/python` on macOS/Linux or
`bootcamp_6\.venv\Scripts\python.exe` on Windows. The browser command above
opens the same saved workbook.

## Worked solutions

Try the exercises first. To open the solutions, run:

```sh
uv run --locked --project bootcamp_6 python bootcamp_8/start.py --solution
```

For VS Code, add both `--setup --solution`, then open
`bootcamp_8/work/solutions.py`. Your exercise answers stay in their own file.

The [exercise template](exercises/workbook.py) and
[solution template](solutions/workbook.py) stay separate from your personal
copies. Setup never overwrites existing answers. Git ignores `work/`; back it
up yourself if needed. To adopt an updated template, rename your personal file
to a backup and run setup again.

Source: adapted from [Hyperparameter Optimization.ipynb](https://github.com/ReLU-NTNU/tutorials-25-26/blob/main/notebooks_2025/bootcamp_7/Hyperparameter%20Optimization.ipynb)
in the private 2025/26 repository. This year's notebook lives here in the
current `education-26-27` repo.
