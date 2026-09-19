# Worked solutions

`solutions.py` is a complete marimo solution notebook for all 27 exercises,
including a two-epoch CPU training loop. It shares the bootcamp's uv environment
and MNIST cache outside the repository. The launcher creates a personal copy
in `work/solutions.py` and preserves it on later launches.

From `bootcamp_2/`, open it with:

```sh
./scripts/start.sh --solution
```

On Windows use `.\scripts\start.ps1 --solution`. The direct command on any OS is:

```sh
uv run --locked python scripts/manage.py start --solution
```

Click **Load / download MNIST**, then **Train for 2 epochs** to run the final
section. Re-running training starts from a fresh copy of the untrained model.

To run every solution and its checks automatically:

```sh
./scripts/test.sh --solution
```

On Windows use `.\scripts\test.ps1 --solution`. Try the student exercises before looking
at these answers.
