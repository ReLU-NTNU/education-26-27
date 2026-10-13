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

    ## Worked solutions

    **All code is complete.** Run all cells to see the results; no edits are
    needed. Read the comments alongside the code and the explanations after each
    task. Try the exercise workbook before using this one to check your answers.

    We fit a baseline forest to handwritten digits, compare settings using
    cross-validation, select a model, and evaluate it on unseen test images.
    The pipeline section is an optional extension.

    The setup below should show sample images and
    **“Split: 1437 training / 360 test”**. The searches fit many forests, so
    allow a few minutes for all results. No GPU or dataset download is needed.
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

    The baseline averages 100 trees, limits their depth to 5, and requires at
    least two examples per leaf. We fit one forest on all training data for the
    eventual test comparison, then cross-validate fresh copies to assess these
    settings. The fitted baseline is not changed by `cross_val_score`.
    """)
    return


@app.cell
def _(RandomForestClassifier, X_train, cross_val_score, np, y_train):
    forest_depth = 5

    # Creating a model sets its options; it has not learned from any images yet.
    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        max_depth=forest_depth,
        min_samples_leaf=2,
        random_state=42,
    )

    # Learn tree split rules from the labelled training images.
    baseline_fitted = rf_baseline.fit(X_train, y_train)

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
    **Worked explanation:** the mean is a **CV validation accuracy**. Every
    score comes from images withheld from that fold's fit, and no test images
    were used. With these settings you should get roughly 0.95 average accuracy.

    Deeper trees can represent more complicated patterns and may reduce bias.
    They can also follow accidental details of their training sample, increasing
    prediction variance. Averaging trees helps reduce that variation. We use
    validation scores to investigate whether changing settings helps in this task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Try every combination with grid search

    The dictionary contains three choices for each of three hyperparameters.
    Grid search tests all 27 combinations using the same five folds. It selects
    by mean validation accuracy and refits the winning settings on all training
    data, ready to make predictions later.
    """)
    return


@app.cell
def _(GridSearchCV, RandomForestClassifier, X_train, pd, y_train):
    depth_options = [5, 10, None]

    leaf_options = [1, 2, 4]

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

    # Fit and validate every combination, then refit the winner on all training rows.
    grid_fitted = grid_search.fit(X_train, y_train)

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
    **Worked explanation:** 3×3×3 = **27 combinations**. Each needs five fold
    fits: 27×5 = 135. The final refit adds one, giving **136 fitted forests**.
    With the supplied split and locked environment, the winning settings are
    **200 trees, no depth limit, and one example per leaf**. Their mean validation
    accuracy is about **0.9763**, compared with the baseline's **0.9471**.

    Sklearn uses `test` to mean the held-out part **inside a CV fold** in this
    column name. It does not refer to our separate test set. The fold-score
    standard deviation describes variation in validation scores; it is not the
    prediction variance at a fixed input from the bias–variance equation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Sample settings with random search

    We sample ten combinations from the distributions below rather than checking
    every possible combination. Each gets the same five-fold validation as the
    grid search. The best sampled setting is then refitted on all training data.
    The random seed fixes the sampled choices so we can reproduce the comparison.
    """)
    return


@app.cell
def _(
    RandomizedSearchCV,
    X_train,
    baseline_cv_scores,
    grid_search,
    pd,
    randint,
    search_forest,
    uniform,
    y_train,
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

    # Set a fixed search budget: ten sampled combinations, each checked on five folds.
    random_trials = 10

    random_search = RandomizedSearchCV(
        estimator=search_forest,
        param_distributions=param_range,
        n_iter=random_trials,
        scoring="accuracy",
        cv=5,
        refit=True,
        random_state=42,
    )

    # Fit the sampled candidates, choose by validation accuracy, and refit the winner.
    random_fitted = random_search.fit(X_train, y_train)

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
    **Worked explanation:** ten candidates need **50 fold fits**, followed by
    one final refit. Another sample could find a better or worse combination.
    With the supplied seed, the winning sample uses **413 trees**, maximum depth
    **12**, minimum split size **4**, minimum leaf size **3**, a feature fraction
    of about **0.316**, and **no bootstrap sampling**. Its mean validation score
    is about **0.9728**, slightly below grid search's **0.9763**.

    This comparison uses different spaces and budgets: grid search tests 27
    combinations, random search samples ten from a broader space. Its result
    does not establish that either search strategy is generally better.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose a model, then check the test set

    We choose the search with the higher mean validation score, using grid search
    to break a tie. Its `best_estimator_` is already trained on all training rows.
    Only now do we predict the test images and compare against their labels.
    """)
    return


@app.cell
def _(
    X_test,
    accuracy_score,
    grid_search,
    pd,
    random_search,
    rf_baseline,
    y_test,
):
    selected_method = (
        "grid"
        if grid_search.best_score_ >= random_search.best_score_
        else "random"
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

    # Predict unseen labels without fitting or changing the already-selected model.
    test_predictions = best_rf.predict(X_test)

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
    **Worked explanation:** CV and test use different images, and the CV models
    train on smaller subsets. Searching also selects candidates that happened
    to perform well on the validation folds.

    We choose **grid search**, because 0.9763 is higher than 0.9728. Its forest
    scores about **0.9639** on the test set, compared with **0.9306** for the
    baseline. That is an improvement of about **3.3 percentage points**. The exact
    winning settings and scores can vary slightly with package versions. A
    small score difference is not proof of a universal advantage from tuning.

    Changing settings after inspecting test accuracy uses the test set to make
    decisions. It then stops being an independent final assessment. Report this
    result honestly; future tuning should use development-only evaluation and
    a fresh untouched test set for a new final assessment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Optional: keep preprocessing inside a pipeline

    The pipeline treats scaling and the forest as a single estimator. Each CV
    fit learns a new scaler from that fold's training rows and applies it to
    the validation rows. Forests do not generally need scaling; the example
    illustrates how to keep fitted preprocessing inside the validation boundary.
    """)
    return


@app.cell
def _(
    Pipeline,
    RandomForestClassifier,
    StandardScaler,
    X_train,
    cross_val_score,
    np,
    y_train,
):
    # The list fixes the order: scale pixel features, then train the classifier.
    # Each pair gives a step a name and an unfitted transformer or model.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42)),
        ],
    )

    pipeline_cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    print("Pipeline validation accuracies:", np.round(pipeline_cv_scores, 4))
    print(f"Average validation accuracy: {pipeline_cv_scores.mean():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Worked explanation:** during one CV fit the scaler uses only that fold's
    training portion; its validation portion does not influence the fitted mean
    or standard deviation. The validation rows are then transformed using those
    training statistics. This applies to learned preprocessing more broadly,
    such as imputation or feature selection.

    A pipeline protects the preprocessing boundary. We still have to choose
    models using validation data and leave test scores out of those decisions.
    """)
    return


if __name__ == "__main__":
    app.run()
