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

## Optional final task: Timestral-5M

The last task compares the existing LSTM, a same-hour-yesterday baseline,
zero-shot Timestral-5M and a locally fine-tuned copy. It uses the first 14 daily
24-hour forecast windows after the existing training split. All models are
scored on the same targets. The last 14 days before the test split are reserved
for validation; optimizer updates only sample earlier windows. The best trained
checkpoint is selected by validation RMSE, with the untouched zero-shot validation
score reported separately. Test scores do not select the checkpoint.

Start it with the optional dependencies:

```sh
uv run --locked --project bootcamp_9 --extra timestral python bootcamp_9/start.py
```

Add `--solution` for the complete version or `--setup` for VS Code. The optional
extra installs the pinned official Timestral Git repository, so Git and internet
access are needed initially. Clicking **Run Timestral task** downloads the
approximately 21 MB pinned checkpoint to `bootcamp_9/.cache/timestral`.
Opening the notebook does not start the optional download or fine-tuning.
Both forecasting and the 40-update fine-tuning exercise run locally on the CPU.
The fine-tuning loop is an educational adapter, not Mistral Forge's training pipeline.

If you already have a personal workbook, rename it as a backup before running
setup to create a fresh copy with this new task. Setup preserves existing files;
you can transfer your earlier exercise answers from the backup.

The original notebook sections are unchanged. The comparison retains each
model's input recipe: the LSTM's original seven lag/time-feature rows, versus
four weeks of consumption history for Timestral. Later forecast origins may
use earlier test observations as history; fine-tuning never updates on them.
This classroom experiment does not establish that AEP was absent from the
foundation model's pretraining data.

Sources: [official Timestral code](https://github.com/mistralai/timestral),
[Timestral-5M model card](https://huggingface.co/mistralai/Timestral-5M).
The checkpoint's weights use Mistral's Modified MIT license; the package code
uses Apache 2.0. Checkpoint and package revisions are pinned in the helper and
uv lockfile respectively.

To check the comparison protocol without downloading weights:

```sh
uv run --locked --project bootcamp_9 python -m unittest discover -s bootcamp_9/tests -v
```
