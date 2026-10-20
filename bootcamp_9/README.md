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
zero-shot Timestral-5M and a locally fine-tuned copy on the same 42 daily
24-hour forecast windows after the existing split. The table reports overall
scores and separate RMSEs for the first two and following four weeks.

Fine-tuning minimizes central point-forecast **MSE**, aligned with the reported
RMSE. It uses two recent years of fitting targets, five full shuffled epochs,
batch size 16, and AdamW (`lr=1e-4`, weight decay `0.01`). The last 12 pre-test
weeks are validation only, with 55 valid daily windows after excluding windows
that cross missing or duplicated hours. Each fitting window is visited once per
epoch. Validation RMSE selects the best trained checkpoint; test scores never
select it. The untouched zero-shot validation baseline and a fixed fitting-loss
curve are shown separately.

Start it with the optional dependencies:

```sh
uv run --locked --project bootcamp_9 --extra timestral python bootcamp_9/start.py
```

Add `--solution` for the complete version or `--setup` for VS Code. The optional
extra installs the pinned official Timestral Git repository, so Git and internet
access are needed initially. Clicking **Run Timestral task** downloads the
approximately 21 MB pinned checkpoint to `bootcamp_9/.cache/timestral`.
Opening the notebook does not start the optional download or fine-tuning.
Forecasting and fine-tuning run locally on the CPU. The public Mistral package
provides inference; the supplied fine-tuning loop is an educational adapter,
not Mistral Forge's training pipeline. It trains the central forecast for squared
error; the other quantiles should not be treated as calibrated intervals after
this exercise.

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

For an optional diagnostic using the actual pretrained checkpoint:

```sh
uv run --locked --project bootcamp_9 --extra timestral python bootcamp_9/tests/check_timestral_learning.py
```

This repeatedly trains on eight fixed fitting windows, verifies nonzero gradients
through every parameter tensor, and checks a substantial decrease in their RMSE.
It does not evaluate validation or test targets. This overfit check establishes
that the training path can learn; validation still decides which checkpoint to use.

To reproduce the selected fine-tuning experiment and its recorded metrics:

```sh
uv run --locked --project bootcamp_9 --extra timestral python bootcamp_9/tests/benchmark_timestral.py
```

The benchmark verifies that pretrained weights remain unchanged, fine-tuned
weights change, and RMSE improves on the combined six weeks and on weeks 3–6.
The recorded CPU run took about 13 seconds for fine-tuning (two threads):

| Period | Zero-shot RMSE (MW) | Fine-tuned RMSE (MW) |
| --- | ---: | ---: |
| Pre-test validation | 723.19 | 666.41 |
| Test weeks 1–2 | 1322.41 | 1345.75 |
| Test weeks 3–6 | 721.30 | 652.37 |
| All six test weeks | 964.24 | 942.02 |

This is a 2.3% overall reduction and a 9.6% reduction in weeks 3–6. Fine-tuning
still worsens the first fortnight by 1.8%; both periods remain visible in the
exercise. The original fortnight had already been inspected during development.
The following four weeks were checked after choosing the recipe on pre-test
validation. These are results for this series and period, not a general guarantee.

[Full selected-run metrics](tests/timestral_benchmark.json) include checkpoint
losses and validation dates. [Validation trials](tests/timestral_validation_trials.json)
record eight network-training recipes (AdamW/RMSprop, point MSE/pinball,
full-network/head-only updates, and different training-history lengths).
Two years of fitting data with full-network AdamW and point MSE won on validation.
Seasonal ridge/boosted residual corrections were also compared using pre-test
validation and did not beat that recipe.
