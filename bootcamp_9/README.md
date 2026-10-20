# Bootcamp 9 · Time series

This is the Time Series notebook we'll use for the lecture.

## Open the class workbook

From the repository folder, run:

```sh
uv run --locked --project bootcamp_9 python bootcamp_9/start.py
```

uv installs the required Python version and packages. The command opens your
personal copy at `bootcamp_9/work/exercises.py` in your browser. Keep the
terminal open while working; press **Ctrl+C** to stop it. Save with
**Ctrl+S / Cmd+S**. The included `AEP_hourly.csv` is loaded automatically.

The original architecture, training, prediction and sequence-building blanks
remain in the class workbook. Replace the `None` placeholders with your code.
Dependent cells wait until those results are available. Run training cells
when you reach them; RNN, LSTM and expanding-window training can take time.

## Solutions

```sh
uv run --locked --project bootcamp_9 python bootcamp_9/start.py --solution
```

This opens a separate personal copy at `bootcamp_9/work/solutions.py`.

## VS Code

Prepare the personal copy without opening the browser:

```sh
uv run --locked --project bootcamp_9 python bootcamp_9/start.py --setup
```

Open `bootcamp_9/work/exercises.py` with **marimo: Open as marimo notebook**.
Select `bootcamp_9/.venv/bin/python` on macOS/Linux or
`bootcamp_9\.venv\Scripts\python.exe` on Windows as the kernel. Install the
Python and marimo extensions if needed. For the solutions, add
`--setup --solution` and open `bootcamp_9/work/solutions.py`.

Setup never overwrites an existing personal copy. Back up `bootcamp_9/work/`
to keep your answers; Git ignores it.

## Source

The class and solutions notebooks and dataset come from
[`tutorials-25-26/notebooks_2026/education_02`](https://github.com/ReLU-NTNU/tutorials-25-26/tree/2a5a9fdb48549b3a620b7ba3e8bfac6f3d20f5f2/notebooks_2026/education_02).
Notebook prose, comments, plots and exercise content are preserved. Variable
names and dependencies are separated between model sections for marimo's
reactive execution; setup uses this folder's locked uv environment.
