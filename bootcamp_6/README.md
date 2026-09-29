# Bootcamp 6 · KNN and decision trees

Learn to classify Iris flowers using nearest neighbours and a single decision
tree. You will fit models, compare their settings and interpret their mistakes.
Allow about **55 minutes**, with two optional extensions at the end.

Choose either **a browser** or **VS Code** below. Both use the same workbook and
save your answers in `bootcamp_6/work/exercises.py`.

## Before you start

Get the repository using the [course instructions](../README.md) and install
[uv](https://docs.astral.sh/uv/getting-started/installation/). You do not need to
install Python separately. The first setup needs internet; the flower dataset
is included, and no GPU is needed.

The commands below start from the **repository folder**, `education-26-27`.

## Option A: work in your browser

Run the command for your computer:

| Computer | Command |
|---|---|
| Mac, including Apple Silicon and Intel | `sh bootcamp_6/scripts/start.sh` |
| Linux | `sh bootcamp_6/scripts/start.sh` |
| Windows, in PowerShell or Command Prompt | `.\bootcamp_6\scripts\start.cmd` |

The first run installs the required software and opens marimo in your browser.
Use the same command to resume later. Keep the terminal open while working;
press **Ctrl+C** in the terminal to stop it. If the browser does not open, follow
the local URL printed in the terminal.

## Option B: work inside VS Code

1. **Prepare the workbook.** From the repository folder, run
   `sh bootcamp_6/scripts/setup.sh` on macOS/Linux or
   `.\bootcamp_6\scripts\setup.cmd` on Windows. This installs the environment and
   creates your personal copy without opening a browser.
2. In VS Code, choose **File → Open Workspace from File…** and open
   [`bootcamp_6.code-workspace`](bootcamp_6.code-workspace) in this folder.
   The window title should include **bootcamp_6 (Workspace)**. Opening the whole
   repository can leave this session's environment out of the kernel picker.
3. Install the recommended **Python** extension from Microsoft and
   [**marimo**](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo).
4. Open **`work/exercises.py`** in the Explorer. If it is hidden, use
   **File → Open File…** and browse to it. Use **marimo: Open as marimo notebook**
   from the Command Palette if it opens as plain Python text.
5. Click the notebook's **kernel picker** at the top right. Choose **marimo** if
   asked for a source, then **relu-bootcamp-six (Python 3.13.x)**. Check that its
   path is inside `bootcamp_6/.venv`. The patch version may differ.
   **Changing Python in the bottom status bar does not switch marimo's kernel.**
6. Click **Run All**. Before starting the exercises, check that the imports run
   without errors and you see an Iris flower table with
   **“Split: 90 train / 30 validation / 30 test”**. Reminders about unfinished
   exercises are expected until you fill in the answers.

### If the project kernel is missing

1. Check the window title: open `bootcamp_6.code-workspace` using step 2 above
   if you are still in the whole repository.
2. In that workspace, choose **Terminal → Run Task… → Bootcamp 6: Setup**.
   Wait for it to finish successfully. This preserves your existing answers.
3. Run **Developer: Reload Window** from the Command Palette, reopen the
   notebook and select its kernel again.
4. If it is still missing, run **Python: Select Interpreter → Enter interpreter
   path…** and browse to the following file, then return to **marimo's kernel
   picker** and select the same environment:

| Computer | Interpreter, relative to `bootcamp_6/` |
|---|---|
| macOS / Linux | `.venv/bin/python` |
| Windows | `.venv\Scripts\python.exe` |

Use the project environment above for this workbook. **marimo sandbox** uses a
separate environment; the other Python versions in the list may not have the
required packages. If you cannot get the project kernel to appear, the browser
command in Option A opens the same saved workbook using the prepared environment.

## Solve and save

Replace `None` beside each **WRITE HERE** comment, then press **Shift+Enter**.
A reminder means an exercise is unfinished; dependent cells wait for its answer.
Read the prompts above each cell and use the results to answer the discussion
questions. Save with **Ctrl+S** / **Cmd+S**.

The seven exercises cover:

1. Predicting a class from a neighbour vote.
2. Implementing KNN: distances, nearest neighbours, labels and a three-class vote.
3. Fitting scikit-learn KNN and predicting validation labels.
4. Comparing different numbers of neighbours.
5. Fitting a small decision tree and reading its diagram.
6. Comparing shallow and deep trees.
7. Evaluating both models on unseen flowers.

The extensions explore feature scaling and the amount of work needed for a
prediction.

## Check the solutions

Try the workbook first. To open the [worked solutions](solutions/workbook.py) in
your browser, add `--solution` to the start command. For VS Code, add `--solution`
to the setup command, then open `work/solutions.py` and select the same `.venv`
kernel. Your exercise answers stay in their own file.

## If something does not work

- **“uv not found”:** install uv using the link above, reopen your terminal and retry.
- **“No module named …” in VS Code:** follow [the project-kernel steps](#if-the-project-kernel-is-missing)
  above. Check the notebook kernel, even if the status bar shows the right Python.
- **You only see Python code:** use **marimo: Open as marimo notebook**.
- **The browser port is occupied:** add `--port 2720` to the start command.

For a setup check, run `sh bootcamp_6/scripts/test.sh` on macOS/Linux or
`.\bootcamp_6\scripts\test.cmd` on Windows from the repository folder. Share the
error message with your instructor if it fails.

Always work in `work/exercises.py`. The files in `exercises/` are the original
templates. Setup and start preserve existing personal files, and your answers
are not uploaded to GitHub. Back up your `work/` folder separately.

To start an updated workbook after a course update, rename your existing
`work/exercises.py` to a backup such as `exercises_previous.py`, then run setup
again. Setup preserves existing files, so it does not replace your saved answers.
