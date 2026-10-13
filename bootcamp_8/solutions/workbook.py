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

    **Worked solutions.** The same random-forest exercises with complete code
    and discussion answers. Try the student version first.

    Adapted from `notebooks_2025/bootcamp_7/Hyperparameter Optimization.ipynb`
    in `ReLU-NTNU/tutorials-25-26`. This keeps its baseline, grid search,
    random search and pipeline flow. Test evaluation is grouped after CV-based
    selection; the numerical digits do not need categorical target encoding.
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
        GridSearchCV,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        StandardScaler,
        accuracy_score,
        cross_val_score,
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
    return X_test, X_train, y_test, y_train


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
def _(RandomForestClassifier, X_train, cross_val_score, mo, np, y_train):
    rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)  # WRITE HERE: create the forest with the specified settings.
    mo.stop(rf_baseline is None, mo.md("Create rf_baseline."))
    baseline_fitted = rf_baseline.fit(X_train, y_train)  # WRITE HERE: rf_baseline.fit on training data.
    mo.stop(baseline_fitted is None, mo.md("Fit the baseline on X_train, y_train."))
    baseline_cv_scores = cross_val_score(rf_baseline, X_train, y_train, cv=5, scoring="accuracy")  # WRITE HERE: cross_val_score with five folds.
    mo.stop(baseline_cv_scores is None, mo.md("Cross-validate the baseline on training data."))
    print("Fold accuracies:", np.round(baseline_cv_scores, 4))
    print(f"Mean CV accuracy: {baseline_cv_scores.mean():.4f}")
    return baseline_cv_scores, rf_baseline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer:** `n_estimators` controls the number of trees, `max_depth` limits
    each tree's depth, and `min_samples_leaf` sets the minimum number of training
    examples in a leaf. These are chosen settings, not learned split locations.
    The seed makes the experiment repeatable. Test data must stay separate from
    the data used to choose settings so we can assess the chosen model fairly.
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
def _(GridSearchCV, RandomForestClassifier, X_train, mo, pd, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 2, 4]}  # WRITE HERE: dictionary with the three lists above.
    mo.stop(param_grid is None, mo.md("Complete param_grid."))
    grid_search = GridSearchCV(rf, param_grid, scoring="accuracy", cv=5)  # WRITE HERE: GridSearchCV with accuracy and five folds.
    mo.stop(grid_search is None, mo.md("Create grid_search."))
    grid_fitted = grid_search.fit(X_train, y_train)  # WRITE HERE: fit the grid search using training data.
    mo.stop(grid_fitted is None, mo.md("Fit grid_search on X_train, y_train."))
    print("Best hyperparameters (grid):", grid_search.best_params_)
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    grid_results = pd.DataFrame(grid_search.cv_results_)[[
        "params", "mean_test_score", "std_test_score",
    ]].sort_values("mean_test_score", ascending=False)
    grid_results
    return grid_search, rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer:** 3×3×3 = 27 settings, 27×5 = 135 fold fits, then one refit.
    `mean_test_score` is the mean CV validation score, not the reserved test
    accuracy. Deeper trees can fit more complex patterns and reduce bias, but
    can become more sensitive to the training sample and increase variance.
    Averaging trees reduces that variation; it does not guarantee zero variance.
    The SD of fold scores is not prediction variance at a fixed input.
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
def _(
    RandomizedSearchCV,
    X_train,
    baseline_cv_scores,
    grid_search,
    mo,
    pd,
    randint,
    rf,
    uniform,
    y_train,
):
    param_range = {
        "n_estimators": randint(50, 501),
        "max_depth": randint(1, 21),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 6),
        "max_features": uniform(0.3, 0.7),
        "bootstrap": [True, False],
    }
    random_search = RandomizedSearchCV(rf, param_distributions=param_range, n_iter=10, cv=5, scoring="accuracy", random_state=42)  # WRITE HERE: RandomizedSearchCV with ten sampled settings.
    mo.stop(random_search is None, mo.md("Create random_search."))
    random_fitted = random_search.fit(X_train, y_train)  # WRITE HERE: fit the random search using training data.
    mo.stop(random_fitted is None, mo.md("Fit random_search on X_train, y_train."))
    print("Best hyperparameters (random):", random_search.best_params_)
    print(f"Best CV accuracy: {random_search.best_score_:.4f}")
    cv_comparison = pd.DataFrame([
        {"Method": "Baseline", "Mean CV accuracy": baseline_cv_scores.mean()},
        {"Method": "Grid search", "Mean CV accuracy": grid_search.best_score_},
        {"Method": "Random search", "Mean CV accuracy": random_search.best_score_},
    ])
    cv_comparison
    return (random_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer:** Ten sampled settings give 50 fold fits and one refit. Sampling
    can explore a large space cheaply, but can miss good settings. These searches
    use different spaces and budgets, so a better score here does not establish
    that either strategy is generally superior. Both use the same five folds.
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
def _(
    X_test,
    accuracy_score,
    grid_search,
    mo,
    pd,
    random_search,
    rf_baseline,
    y_test,
):
    chosen_search = grid_search if grid_search.best_score_ >= random_search.best_score_ else random_search  # WRITE HERE: select by best CV score, grid wins ties.
    mo.stop(chosen_search is None, mo.md("Choose a search using CV scores only."))
    best_rf = chosen_search.best_estimator_
    test_predictions = best_rf.predict(X_test)  # WRITE HERE: best_rf.predict on X_test, after choosing.
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
    **Answer:** The test and CV folds contain different examples, and CV trains
    on smaller subsets. Selection also favors settings that happened to score
    well in CV. Report the chosen model's test result without tuning against it;
    otherwise this becomes another validation set rather than a final check.
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
def _(
    Pipeline,
    RandomForestClassifier,
    StandardScaler,
    X_train,
    cross_val_score,
    mo,
    np,
    y_train,
):
    pipeline = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(random_state=42))])  # WRITE HERE: the two-step Pipeline described above.
    mo.stop(pipeline is None, mo.md("Build the scaler-and-forest pipeline."))
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")  # WRITE HERE: cross_val_score on the whole pipeline, training data only.
    mo.stop(scores is None, mo.md("Cross-validate the entire pipeline."))
    print("Pipeline fold accuracies:", np.round(scores, 4))
    print(f"Mean CV accuracy: {scores.mean():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer:** Each fold fits a new scaler using only its training rows, then
    transforms the validation rows with those same statistics. Fitting the
    scaler before CV includes information from validation rows. A pipeline
    protects preprocessing boundaries, but cannot prevent test-driven choices.
    """)
    return


if __name__ == "__main__":
    app.run()
