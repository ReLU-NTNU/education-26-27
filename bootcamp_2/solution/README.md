# Worked solutions

`solutions.py` is a complete marimo solution notebook for all 27 exercises,
including a two-epoch CPU training loop. It shares the bootcamp's uv environment
and MNIST cache.

From `bootcamp_2/`, open it with:

```sh
./start.sh --solution
```

On Windows use `.\start.ps1 --solution`. The direct command on any OS is:

```sh
uv run --locked python lecture.py start --solution
```

Click **Load / download MNIST**, then **Train for 2 epochs** to run the final
section. Re-running training starts from a fresh copy of the untrained model.

To run every solution and its checks automatically:

```sh
./test.sh --solution
```

On Windows use `.\test.ps1 --solution`. Try the student exercises before looking
at these answers.
