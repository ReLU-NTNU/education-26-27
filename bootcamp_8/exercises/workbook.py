# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "marimo==0.24.2",
#     "matplotlib>=3.10,<4",
#     "numpy>=2.3,<3",
#     "pandas>=2.3,<4",
#     "scikit-learn>=1.7,<2",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootcamp 8 · Hyperparameter optimization

    **Student exercises.** This is the previous random-forest demo turned into
    exercises: baseline → grid search → random search → pipeline.

    Replace `None` at each **WRITE HERE** comment, press **Shift+Enter**, and
    answer the discussion questions. Unfinished cells show reminders and dependent
    cells wait. In marimo, edit the code inside the cells; the editor manages
    the outer functions. Save with **Ctrl+S / Cmd+S**.

    We use the built-in handwritten digits: 64 pixel features, labels 0–9.
    The data loading and split are provided. No dataset download or GPU is needed.
    Use only training data for the searches; inspect the reserved test results
    after selecting a model by cross-validation.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from scipy.stats import randint, uniform
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        RandomForestClassifier,
        accuracy_score,
        load_digits,
        np,
        pd,
        randint,
        train_test_split,
        uniform,
    )


@app.cell
def _(load_digits, pd, train_test_split):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f"Split: {len(y_train)} training / {len(y_test)} test")
    pd.DataFrame(X_train[:5]).assign(digit=y_train[:5])
    return X_test, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Fit a baseline random forest

    Create a `RandomForestClassifier` with `n_estimators=100`, `max_depth=5`,
    `min_samples_leaf=2`, and `random_state=42`. Fit it on `X_train, y_train`.

    Then use `cross_val_score` with the baseline, `X_train`, `y_train`, `cv=5`
    and `scoring="accuracy"`. Inspect the five scores and their average.
    `cross_val_score` fits a fresh copy for each fold; the fitted baseline will
    also be used for the final test comparison.

    **Discuss:** Which settings are hyperparameters? What does each one control?
    Why do we keep the test data out of cross-validation?
    """)
    return


@app.cell
def _(mo, np):
    rf_baseline = None  # WRITE HERE: create the forest with the specified settings.
    mo.stop(rf_baseline is None, mo.md("Create rf_baseline."))
    baseline_fitted = None  # WRITE HERE: rf_baseline.fit on training data.
    mo.stop(baseline_fitted is None, mo.md("Fit the baseline on X_train, y_train."))
    baseline_cv_scores = None  # WRITE HERE: cross_val_score with five folds.
    mo.stop(baseline_cv_scores is None, mo.md("Cross-validate the baseline on training data."))
    print("Fold accuracies:", np.round(baseline_cv_scores, 4))
    print(f"Mean CV accuracy: {baseline_cv_scores.mean():.4f}")
    return baseline_cv_scores, rf_baseline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer:** explain the settings and why the test data stays separate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Grid search

    The unfitted forest `rf` is provided. Complete the search grid:

    - `n_estimators`: **[100, 200, 300]**
    - `max_depth`: **[5, 10, None]**
    - `min_samples_leaf`: **[1, 2, 4]**

    Here `None` inside the depth list means no depth limit.
    Create `GridSearchCV` with `rf`, `param_grid`, `scoring="accuracy"`, and
    `cv=5`. Fit it on training data. Keep the default `refit=True`: after CV,
    it fits the best settings on all training data.

    **Before running:** how many settings and fold fits are there? The original
    grid has 27 settings and can take a few minutes. If you need a shorter run,
    use `[100]` for `n_estimators` and state that you reduced the search.

    **Discuss:** Does `mean_test_score` in `cv_results_` refer to our reserved
    `X_test`? What effect can tree depth have on bias and variance?
    """)
    return


@app.cell
def _(RandomForestClassifier, mo, pd):
    rf = RandomForestClassifier(random_state=42)
    param_grid = None  # WRITE HERE: dictionary with the three lists above.
    mo.stop(param_grid is None, mo.md("Complete param_grid."))
    grid_search = None  # WRITE HERE: GridSearchCV with accuracy and five folds.
    mo.stop(grid_search is None, mo.md("Create grid_search."))
    grid_fitted = None  # WRITE HERE: fit the grid search using training data.
    mo.stop(grid_fitted is None, mo.md("Fit grid_search on X_train, y_train."))
    print("Best hyperparameters (grid):", grid_search.best_params_)
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    grid_results = pd.DataFrame(grid_search.cv_results_)[[
        "params", "mean_test_score", "std_test_score",
    ]].sort_values("mean_test_score", ascending=False)
    grid_results
    return (grid_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer:** explain the fitting budget, the score name, and tree depth.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Random search

    The original search distributions are provided. `randint(a, b)` draws
    integers from `a` through `b-1`; `uniform(0.3, 0.7)` draws feature fractions
    between 0.3 and 1.0.

    Create `RandomizedSearchCV` using `rf`, `param_distributions=param_range`,
    `n_iter=10`, `cv=5`, `scoring="accuracy"`, and `random_state=42`.
    Fit it on training data and compare the best CV scores.

    **Discuss:** How many fold fits does random search use? Why might it miss
    the best setting? Is this a fair comparison of the two search strategies?
    """)
    return


@app.cell
def _(baseline_cv_scores, grid_search, mo, pd, randint, uniform):
    param_range = {
        "n_estimators": randint(50, 501),
        "max_depth": randint(1, 21),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 6),
        "max_features": uniform(0.3, 0.7),
        "bootstrap": [True, False],
    }
    random_search = None  # WRITE HERE: RandomizedSearchCV with ten sampled settings.
    mo.stop(random_search is None, mo.md("Create random_search."))
    random_fitted = None  # WRITE HERE: fit the random search using training data.
    mo.stop(random_fitted is None, mo.md("Fit random_search on X_train, y_train."))
    print("Best hyperparameters (random):", random_search.best_params_)
    print(f"Best CV accuracy: {random_search.best_score_:.4f}")
    cv_comparison = pd.DataFrame([
        {"Method": "Baseline", "Mean CV accuracy": baseline_cv_scores.mean()},
        {"Method": "Grid search", "Mean CV accuracy": grid_search.best_score_},
        {"Method": "Random search", "Mean CV accuracy": random_search.best_score_},
    ])
    cv_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer:** compare the fitting budgets and explain the tradeoff.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use the test set after choosing the model

    Choose the search with the higher `best_score_`; use grid search for a tie.
    Complete `chosen_search`, and predict the reserved test data using its
    `best_estimator_`. Both searches already refit their winner on all training
    data. The supplied code compares it with the fitted baseline.

    **Before filling in the predictions:** record your chosen settings and why
    you selected them. Inspect these final test accuracies once; do not change
    the model or search settings in response to the test result.
    """)
    return


@app.cell
def _(X_test, accuracy_score, mo, pd, rf_baseline, y_test):
    chosen_search = None  # WRITE HERE: select by best CV score, grid wins ties.
    mo.stop(chosen_search is None, mo.md("Choose a search using CV scores only."))
    best_rf = chosen_search.best_estimator_
    test_predictions = None  # WRITE HERE: best_rf.predict on X_test, after choosing.
    mo.stop(test_predictions is None, mo.md("Record your choice, then predict X_test."))
    test_results = pd.DataFrame([
        {"Model": "Baseline", "Test accuracy": rf_baseline.score(X_test, y_test)},
        {"Model": "Chosen tuned forest", "Test accuracy": accuracy_score(y_test, test_predictions)},
    ])
    test_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer:** why can the test score differ from the best CV score?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · A pipeline

    Build a `Pipeline` with `("scaler", StandardScaler())` followed by
    `("rf", RandomForestClassifier(random_state=42))`. Cross-validate the
    **whole pipeline** on `X_train, y_train`, with `cv=5` and accuracy scoring.
    Keep this example separate from the model selected above; do not test it.

    This keeps the original demo's scaler-and-forest structure. Forests do not
    generally need scaling; the purpose is to learn how a pipeline keeps fitted
    preprocessing inside each training fold. Our pixel features are numerical,
    so there is no categorical target encoder in this example.

    **Discuss:** Why is fitting a scaler on all data before cross-validation
    leakage? Does the pipeline make it okay to choose settings using test scores?
    """)
    return


@app.cell
def _(mo, np):
    pipeline = None  # WRITE HERE: the two-step Pipeline described above.
    mo.stop(pipeline is None, mo.md("Build the scaler-and-forest pipeline."))
    scores = None  # WRITE HERE: cross_val_score on the whole pipeline, training data only.
    mo.stop(scores is None, mo.md("Cross-validate the entire pipeline."))
    print("Pipeline fold accuracies:", np.round(scores, 4))
    print(f"Mean CV accuracy: {scores.mean():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer:** explain what gets fitted inside a fold and what remains your responsibility.
    """)
    return


if __name__ == "__main__":
    app.run()
