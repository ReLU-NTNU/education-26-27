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
    # Bootcamp 8 · Choosing settings for a random forest

    Train a model to recognise handwritten digits, then investigate whether
    changing its settings improves its predictions on unseen images.

    You will fit a baseline forest, compare settings with cross-validation,
    try grid and random search, and evaluate your chosen model. The final
    pipeline exercise is an optional extension.

    ### How to use this workbook

    1. Run the provided import and data cells. You should see sample images and
       **“Split: 1437 training / 360 test”**.
    2. Read a task, then replace only the `None` values under **WRITE HERE**.
       Keep the variable names and the reminder lines unchanged.
    3. Press **Shift+Enter** to run the cell. If it still shows a reminder, finish
       its remaining blanks. Later cells wait for the earlier answers.
    4. Read the output and answer the discussion questions in your own notes.
       Save with **Ctrl+S / Cmd+S**.

    `None` in an answer means “not filled in yet”. You may also see `None` in a
    list of tree depths: there it is a valid setting meaning **no depth limit**.

    In the marimo editor, edit inside the notebook cells. You do not need to
    edit the outer `def ...` or `return ...` lines in the saved Python file.
    The data loading, result tables and search setup are provided so you can
    concentrate on the model choices. No GPU or dataset download is needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the code

    | Code | Meaning |
    |---|---|
    | `model = RandomForestClassifier(...)` | Create an unfitted model with chosen settings. |
    | `model.fit(X_train, y_train)` | Learn from the training features and their correct labels. |
    | `model.predict(X_new)` | Predict labels for new feature rows. |
    | `accuracy_score(correct_labels, predictions)` | Fraction of labels predicted correctly: 0.9 means 90%. |
    | `scores.mean()` | Average the numbers in an array of scores. |

    A **hyperparameter** is a setting we choose, such as maximum tree depth.
    The tree's split rules are learned during `.fit()`.

    The imports and data preparation below are provided. Run them as they are.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # These distributions describe which settings random search may sample.
    from scipy.stats import randint, uniform

    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        cross_val_score,
        train_test_split,
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
        plt,
        randint,
        train_test_split,
        uniform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What are X and y?

    One example is an **8×8 image of a handwritten digit**. Each pixel contains
    a brightness value between 0 and 16. Flattening the image gives 64 input
    features, so one row of `X` contains 64 numbers. Its matching entry in `y`
    is the correct digit, from 0 to 9.

    We split the examples before fitting anything:

    - **Training set:** learn the model and choose its settings using CV.
    - **Test set:** check the chosen model at the end. Do not use it in searches.

    `stratify=y` keeps similar digit proportions in both sets. `random_state=42`
    makes the split repeatable, so your results can be compared with the solutions.
    """)
    return


@app.cell
def _(load_digits, pd, train_test_split):
    # Load all pixel rows and the corresponding correct digit labels.
    X, y = load_digits(return_X_y=True)

    # Reserve 20% of the examples for the final test; keep the other 80% for training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"Split: {len(y_train)} training / {len(y_test)} test")
    print(f"Each image has {X_train.shape[1]} pixel features.")

    # Show the first eight pixels of five images, followed by their correct labels.
    data_preview = pd.DataFrame(
        X_train[:5, :8],
        columns=[f"Pixel {_pixel}" for _pixel in range(8)],
    )
    data_preview["Correct digit"] = y_train[:5]
    data_preview
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train, plt, y_train):
    # Reshape four training rows back into images to see what the model receives.
    digit_figure, _axes = plt.subplots(1, 4, figsize=(8, 2))
    for _axis, _pixels, _label in zip(_axes, X_train[:4], y_train[:4]):
        _axis.imshow(_pixels.reshape(8, 8), cmap="gray", vmin=0, vmax=16)
        _axis.set_title(f"Correct digit: {_label}")
        _axis.axis("off")
    digit_figure.tight_layout()
    digit_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How five-fold cross-validation works

    Split the **training set** into five groups. Fit a fresh model on four groups
    and score it on the fifth. Repeat until every group has taken a turn as the
    validation group. This gives five validation accuracies; we compare their mean.

    `cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")` does
    those fits and returns the five scores. It fits fresh copies, so you do not
    need to fit the model first to cross-validate it. With this classifier, the
    folds approximately preserve the class proportions.

    **The test set is not one of these five groups.** It stays separate throughout.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Train a baseline forest

    A **baseline** is our starting model. It gives us something to compare with
    after tuning. The forest below uses:

    - `n_estimators=100`: average the votes of 100 trees.
    - `max_depth=5`: each tree can grow at most five levels below its root.
    - `min_samples_leaf=2`: each final leaf needs at least two training examples.

    **Your task:** set `forest_depth` to **5**, then complete
    `baseline_fitted` with `rf_baseline.fit(X_train, y_train)`.
    The validation call and output are provided. Run the cell and look for five
    accuracies and their mean. For example, an accuracy of `0.95` means 95% correct.

    **Discuss:** what might happen if we increase the depth? Consider both
    the patterns a tree can learn and its sensitivity to the training sample.
    """)
    return


@app.cell
def _(RandomForestClassifier, X_train, cross_val_score, mo, np, y_train):
    # WRITE HERE: the starting maximum tree depth is 5.
    forest_depth = None
    mo.stop(
        forest_depth is None,
        mo.md('Set forest_depth to 5, then run this cell again.'),
    )

    # Creating a model sets its options; it has not learned from any images yet.
    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        max_depth=forest_depth,
        min_samples_leaf=2,
        random_state=42,
    )

    # WRITE HERE: fit this forest on the training pixel rows and correct labels.
    baseline_fitted = None
    mo.stop(
        baseline_fitted is None,
        mo.md('Use rf_baseline.fit(X_train, y_train).'),
    )

    # Each fold fits a new forest and measures accuracy on that fold's validation rows.
    baseline_cv_scores = cross_val_score(
        rf_baseline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    print("Five validation accuracies:", np.round(baseline_cv_scores, 4))
    print(f"Average validation accuracy: {baseline_cv_scores.mean():.4f}")
    return baseline_cv_scores, rf_baseline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check your understanding:** is the average above a training accuracy,
    a CV validation accuracy, or our final test accuracy? Explain why.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Try every combination with grid search

    A grid is a **dictionary of setting names and lists of possible values**.
    For example, `{"max_depth": [2, 4]}` asks a search to try both depth limits.
    If another setting has three choices, the search tries 2×3 = 6 combinations.

    **Your task:** replace `depth_options` with **[5, 10, None]** and
    `leaf_options` with **[1, 2, 4]**. The tree-count choices are already supplied.
    Complete `grid_fitted` with `grid_search.fit(X_train, y_train)`.

    `GridSearchCV` tests each combination with five-fold CV, picks the highest
    average validation accuracy, and refits that winner on all training data.
    This last step is controlled by `refit=True`.

    **Before running:** predict the number of combinations and fold fits.
    This grid can take a few minutes. For a shorter run, change the provided
    tree-count list to `[100]` and note that you used a smaller grid.

    **Read the output:** `best_params_` gives the winning settings.
    `best_score_` is their average validation accuracy. The table lists every
    combination, with the best scores at the top.
    """)
    return


@app.cell
def _(GridSearchCV, RandomForestClassifier, mo, pd):
    # WRITE HERE: list the three maximum-depth choices: 5, 10, and no depth limit.
    depth_options = None
    mo.stop(
        depth_options is None,
        mo.md('Set depth_options to [5, 10, None].'),
    )

    # WRITE HERE: list the three minimum-leaf-size choices: 1, 2, and 4.
    leaf_options = None
    mo.stop(
        leaf_options is None,
        mo.md('Set leaf_options to [1, 2, 4].'),
    )

    # Each dictionary key names a forest setting; its list gives the choices to try.
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": depth_options,
        "min_samples_leaf": leaf_options,
    }

    # This unfitted forest is a template: the search creates a fresh copy for each fit.
    search_forest = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=search_forest,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        refit=True,
    )

    # WRITE HERE: run the search using training data only.
    grid_fitted = None
    mo.stop(
        grid_fitted is None,
        mo.md('Use grid_search.fit(X_train, y_train).'),
    )

    print("Best grid settings:")
    for _setting, _value in grid_search.best_params_.items():
        print(f"  {_setting}: {_value}")
    print(f"Best average validation accuracy: {grid_search.best_score_:.4f}")

    # Give sklearn's CV result columns clearer names for the displayed table.
    grid_results = pd.DataFrame(grid_search.cv_results_)[
        ["params", "mean_test_score", "std_test_score"]
    ].rename(columns={
        "params": "Settings",
        "mean_test_score": "Mean validation accuracy",
        "std_test_score": "Variation between fold scores",
    })
    grid_results.sort_values("Mean validation accuracy", ascending=False)
    return grid_search, search_forest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check your understanding:**

    1. How many forests did the grid fit, including the final refit?
    2. Which depth and leaf size won? How does their CV score compare with the baseline?
    3. Why does sklearn call the column `mean_test_score` when we have not used `X_test`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Sample settings with random search

    Grid search grows expensive when many settings have many choices. Random
    search tries a fixed number of sampled combinations instead.

    The distributions below are provided. `randint(50, 501)` samples a whole
    number from **50 through 500**. `uniform(0.3, 0.7)` samples a fraction from
    **0.3 up to 1.0**: the second number is the range width, not its upper limit.

    **Your task:** set `random_trials` to **10**, then complete `random_fitted`
    with `random_search.fit(X_train, y_train)`.
    `n_iter=random_trials` sets the number of combinations to sample.
    Each still gets five-fold CV. The seed makes sampling repeatable.

    Read the comparison table. A higher average validation accuracy is better.
    Random search is not guaranteed to beat grid search.
    """)
    return


@app.cell
def _(
    RandomizedSearchCV,
    baseline_cv_scores,
    grid_search,
    mo,
    pd,
    randint,
    search_forest,
    uniform,
):
    # These ranges define the settings random search is allowed to try.
    param_range = {
        "n_estimators": randint(50, 501),     # Between 50 and 500 trees.
        "max_depth": randint(1, 21),          # Maximum depth between 1 and 20.
        "min_samples_split": randint(2, 11),  # At least 2–10 examples to split a node.
        "min_samples_leaf": randint(1, 6),    # At least 1–5 examples in a leaf.
        "max_features": uniform(0.3, 0.7),    # Fraction of features considered at a split.
        "bootstrap": [True, False],          # Bootstrap each tree's rows, or use all training rows.
    }

    # WRITE HERE: try ten sampled combinations.
    random_trials = None
    mo.stop(
        random_trials is None,
        mo.md('Set random_trials to 10.'),
    )

    random_search = RandomizedSearchCV(
        estimator=search_forest,
        param_distributions=param_range,
        n_iter=random_trials,
        scoring="accuracy",
        cv=5,
        refit=True,
        random_state=42,
    )

    # WRITE HERE: run random search on the same training data as grid search.
    random_fitted = None
    mo.stop(
        random_fitted is None,
        mo.md('Use random_search.fit(X_train, y_train).'),
    )

    print("Best random-search settings:")
    for _setting, _value in random_search.best_params_.items():
        print(f"  {_setting}: {_value}")
    print(f"Best average validation accuracy: {random_search.best_score_:.4f}")

    # Compare methods using validation scores; the final test data is still untouched.
    cv_comparison = pd.DataFrame([
        {
            "Method": "Baseline",
            "Mean validation accuracy": baseline_cv_scores.mean(),
        },
        {
            "Method": "Grid search",
            "Mean validation accuracy": grid_search.best_score_,
        },
        {
            "Method": "Random search",
            "Mean validation accuracy": random_search.best_score_,
        },
    ])
    cv_comparison
    return (random_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check your understanding:** how many fold fits did this search use?
    Could another random sample of settings give a different winner? Does this
    comparison prove that one search strategy is always better?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose a model, then check the test set

    Look at the CV comparison table. Which **search** has the higher validation
    score? Set `selected_method` to **"grid"** or **"random"**, using quotes.
    If the scores tie, choose **"grid"**. Write down the winning settings and
    your reason before looking at the test result.

    The search's `best_estimator_` is its winning forest, already fitted on all
    training images because we used `refit=True`. No new `.fit()` is needed.

    **Your task:** complete `test_predictions` with `best_rf.predict(X_test)`.
    The provided code compares those predicted digits with `y_test` and reports
    the accuracy of the baseline and selected model.

    After viewing this final score, report it. Changing settings in response to
    it would turn the test set into another validation set.
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
    # WRITE HERE: choose "grid" or "random" by comparing their validation scores.
    selected_method = None
    mo.stop(
        selected_method not in ('grid', 'random'),
        mo.md('Set selected_method to "grid" or "random" using the CV table.'),
    )

    # Look up the already-refitted winning forest from the selected search.
    if selected_method == "grid":
        chosen_search = grid_search
    else:
        chosen_search = random_search
    best_rf = chosen_search.best_estimator_

    print("Chosen settings:")
    for _setting, _value in chosen_search.best_params_.items():
        print(f"  {_setting}: {_value}")

    # WRITE HERE: predict digit labels from the reserved test pixel rows.
    test_predictions = None
    mo.stop(
        test_predictions is None,
        mo.md('Use best_rf.predict(X_test) after recording your choice.'),
    )

    # Calculate the fraction of correct labels on the separate test set.
    tuned_test_accuracy = accuracy_score(y_test, test_predictions)
    baseline_test_accuracy = rf_baseline.score(X_test, y_test)
    test_results = pd.DataFrame([
        {"Model": "Baseline", "Test accuracy": baseline_test_accuracy},
        {"Model": "Chosen tuned forest", "Test accuracy": tuned_test_accuracy},
    ])
    test_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check your understanding:** why might test accuracy differ from CV
    accuracy? If your test score is disappointing, why should you not change
    your choice using that score?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Optional: keep preprocessing inside a pipeline

    A **pipeline** bundles several steps into one model. Here we first scale
    feature values, then fit a forest. Scaling means subtracting the training
    mean and dividing by the training standard deviation for each feature.

    Forests generally do not need this scaling. We use this small example to
    learn where preprocessing must be fitted. In a CV pipeline, a fresh scaler
    learns its statistics from **that fold's training rows only**, then applies
    them to the validation rows. Scaling everything before CV would include
    information from validation rows.

    The two-step pipeline is provided. **Your task:** complete `pipeline_cv_scores`
    with a `cross_val_score` call. Use `pipeline`, `X_train`, `y_train`, `cv=5`,
    and `scoring="accuracy"`, just as in Exercise 1.
    Keep this illustration separate from your selected forest; do not test it.
    """)
    return


@app.cell
def _(Pipeline, RandomForestClassifier, StandardScaler, mo, np):
    # The list fixes the order: scale pixel features, then train the classifier.
    # Each pair gives a step a name and an unfitted transformer or model.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42)),
        ],
    )

    # WRITE HERE: cross-validate the whole pipeline, not just its forest step.
    pipeline_cv_scores = None
    mo.stop(
        pipeline_cv_scores is None,
        mo.md('Use cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").'),
    )

    print("Pipeline validation accuracies:", np.round(pipeline_cv_scores, 4))
    print(f"Average validation accuracy: {pipeline_cv_scores.mean():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check your understanding:** which rows does the scaler learn from during
    one CV fit? Does a pipeline prevent us from making choices based on test scores?
    """)
    return


if __name__ == "__main__":
    app.run()
