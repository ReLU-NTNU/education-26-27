# Bootcamp 9 · Time series

Open the class workbook from the repository folder:

```sh
uv run --locked --project bootcamp_9 python bootcamp_9/start.py
```

For the worked solutions:

```sh
uv run --locked --project bootcamp_9 python bootcamp_9/start.py --solution
```

uv installs Python 3.13 and all lecture dependencies, including Timestral. Git
and internet access are needed for the first installation. Setup instructions
are also included at the beginning of both workbooks.

Your answers are saved in `bootcamp_9/work/exercises.py`; solutions use
`bootcamp_9/work/solutions.py`. The launcher reopens these personal copies
without overwriting them. Save with **Ctrl+S / Cmd+S** and keep the terminal
open while working. Back up `work/` to keep a separate copy of your answers.

For VS Code, add `--setup` to either command, open the printed file as a marimo
notebook, and select `bootcamp_9/.venv/bin/python` as the kernel
(Windows: `bootcamp_9/.venv/Scripts/python.exe`).

## Files

- `exercises/workbook.py` — class workbook, including the final Timestral task.
- `solutions/workbook.py` — worked solutions.
- `helpers.py` — shared forecasting and training utilities.
- `AEP_hourly.csv` — the lecture dataset.
- `start.py` — launcher; `pyproject.toml`, `uv.lock` and `.python-version` define the environment.
- `work/`, `.venv/` and `.cache/` — generated local files, ignored by Git.

The final task compares the LSTM, a seasonal baseline, zero-shot Timestral-5M
and a fine-tuned copy on the same daily forecasts. Its run button starts the
approximately 21 MB checkpoint download and CPU fine-tuning. The workbooks
contain the experiment instructions and results table, full-period and day/week
forecast views, daily RMSE, and fitting/validation curves.

Maintainer checks live outside the lesson folder:

```sh
uv run --locked --project bootcamp_9 python -m unittest discover -s tests -v
```

## Sources

The original lecture content and dataset come from
[`tutorials-25-26/notebooks_2026/education_02`](https://github.com/ReLU-NTNU/tutorials-25-26/tree/2a5a9fdb48549b3a620b7ba3e8bfac6f3d20f5f2/notebooks_2026/education_02).

Timestral uses the pinned [official package](https://github.com/mistralai/timestral)
and [Timestral-5M checkpoint](https://huggingface.co/mistralai/Timestral-5M).
The code uses Apache 2.0; the weights use Mistral's Modified MIT license.
