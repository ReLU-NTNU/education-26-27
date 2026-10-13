# Bootcamp 8 · Choosing settings for a random forest

Learn to recognise handwritten digits with a random forest and choose its
settings using cross-validation. The workbook explains the data, the model
methods and the validation procedure before you begin the exercises.

1. Fit and cross-validate a baseline forest on handwritten digits.
2. Tune the forest with grid search.
3. Compare a random search.
4. Optionally cross-validate a preprocessing pipeline.

Each task has small `WRITE HERE` blanks, hints and discussion questions. Data
preparation, search setup and result tables are provided. Choose the tuned
model using CV scores before the final test comparison. The grid has 135 fold
fits, so allow a few minutes for it; a smaller grid is suggested in the prompt
if needed. The data is included with scikit-learn; no GPU is needed.

## Open the exercises

Get the repository and install **uv** using the [course instructions](../README.md).
From the repository folder, run this command on macOS, Linux or Windows:

```sh
uv run --locked --project bootcamp_6 python bootcamp_8/start.py
```

The command opens a personal copy at `bootcamp_8/work/exercises.py` in your browser.
Keep the terminal open while
working; press **Ctrl+C** to stop it. If the browser does not open, follow the
local URL printed in the terminal.

Replace `None` beside each **WRITE HERE** comment and press **Shift+Enter**.
Hints for unfinished exercises are expected; dependent cells wait. The provided
setup should show four digit images and **“Split: 1437 training / 360 test”**.
Save with **Ctrl+S / Cmd+S**. Keep your discussion answers in your own notes.

## Use VS Code instead

First prepare the personal copy without opening a browser:

```sh
uv run --locked --project bootcamp_6 python bootcamp_8/start.py --setup
```

Open the repo in VS Code and open `bootcamp_8/work/exercises.py`. Use **marimo:
Open as marimo notebook**, and select the Python kernel in `bootcamp_6/.venv`.
The environment is named **relu-bootcamp-six (Python 3.13.x)**. Install
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
`bootcamp_8/work/solutions.py`. All solution cells are complete: use **Run All**
without editing them, and read the worked explanations after each section.
Your exercise answers stay in their own file.

The [exercise template](exercises/workbook.py) and
[solution template](solutions/workbook.py) stay separate from your personal
copies. Setup never overwrites existing answers. Git ignores `work/`; back it
up yourself if needed. To adopt an updated template, rename your personal file
to a backup and run setup again.
