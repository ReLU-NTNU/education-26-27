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
    # Bootcamp 6 · Neighbours and decision trees

    **Worked solutions.** Try the exercise notebook before reading these answers.

    By the end, you will be able to make a neighbour vote, fit KNN and a single
    decision tree, choose model settings using validation data, and read a confusion matrix.

    **How to work:** replace `None` at each `WRITE HERE` comment, then press
    **Shift+Enter**. Unfinished cells show a hint and dependent cells wait. Save with
    **Ctrl+S / Cmd+S**. Variables beginning with `_` are local to a cell; define other
    variable names in only one cell. Work through Exercises 1–6; the last two are short extensions.

    No forest, boosting, GPU or dataset download is needed. The Iris data comes with scikit-learn.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
    return ConfusionMatrixDisplay, DecisionTreeClassifier, KNeighborsClassifier, StandardScaler, accuracy_score, classification_report, load_iris, make_pipeline, np, pd, plot_tree, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Meet the data (provided)

    Each row is a flower. Four **features** describe sepal and petal length/width in centimetres.
    The **target** is its species: 0 = setosa, 1 = versicolor, 2 = virginica.

    We split the 150 flowers into **90 training**, **30 validation**, and **30 test** examples.
    Training data fits the models. Validation data helps choose `k` and tree depth.
    Keep the test set for the final comparison in Exercise 6. `stratify` keeps class proportions
    balanced; `random_state` makes the split repeatable.
    """)
    return


@app.cell
def _(load_iris, mo, pd, train_test_split):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    _preview = pd.DataFrame(X_train[:6], columns=iris.feature_names)
    _preview["species"] = iris.target_names[y_train[:6]]
    mo.vstack([mo.md(f"**Split:** {len(y_train)} train / {len(y_val)} validation / {len(y_test)} test"), _preview])
    return X_test, X_train, X_trainval, X_val, iris, y_test, y_train, y_trainval, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · A neighbour vote (3 minutes)

    KNN predicts a class by finding the closest training examples and taking a vote.
    For this first question, no code library is needed.

    The three closest neighbours have labels **[versicolor, virginica, versicolor]**.
    Set `neighbour_vote` to the winning class name (a string).
    """)
    return


@app.cell
def _(mo):
    neighbour_vote = "versicolor"
    mo.stop(neighbour_vote is None, mo.md("✏️ Replace None with the class receiving the most votes."))
    assert neighbour_vote == "versicolor", "Count how often each species appears."
    mo.md("✅ Two out of three neighbours vote for **versicolor**.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Fit your first KNN model (7 minutes)

    `fit(X, y)` learns from labelled training examples; `predict(X)` predicts labels for new examples.
    `accuracy_score` is the fraction of predictions that are correct.

    1. Set `k_start` to **5**.
    2. Fit the provided pipeline using `X_train` and `y_train`.
    3. Predict the validation labels using `knn_model.predict(X_val)`.

    **Why scale?** KNN measures distance. `StandardScaler` puts each feature on a comparable
    scale using the training mean and standard deviation. Keeping it inside a pipeline means
    the same training transformation is used on validation and test data, without fitting on them.
    """)
    return


@app.cell
def _(KNeighborsClassifier, StandardScaler, X_train, X_val, accuracy_score, make_pipeline, mo, np, y_train, y_val):
    k_start = 5
    knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_start))
    knn_fitted = knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_val)
    mo.stop(knn_predictions is None, mo.md("✏️ Call predict with the validation features."))
    assert k_start == 5, "Use five neighbours for this starting model."
    assert np.asarray(knn_predictions).shape == y_val.shape, "Return one prediction per validation flower."
    assert np.array_equal(knn_predictions, knn_model.predict(X_val)), "Use this model's validation predictions."
    mo.md(f"✅ KNN validation accuracy: **{accuracy_score(y_val, knn_predictions):.1%}**")
    return (knn_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · What changes when k changes? (6 minutes)

    Set `k_values` to **[1, 5, 15, 30]** and run the provided experiment.
    The table shows training and validation accuracy, not test accuracy.

    Compare the two columns. Does the model that best fits training data also do best on validation?
    A small `k` can follow very local details; a larger `k` averages over a wider neighbourhood.
    A perfect training score is not evidence of perfect performance on new flowers.
    """)
    return


@app.cell
def _(KNeighborsClassifier, StandardScaler, X_train, X_val, make_pipeline, mo, pd, y_train, y_val):
    k_values = [1, 5, 15, 30]
    mo.stop(k_values is None, mo.md("✏️ Enter the four k values as a list."))
    assert k_values == [1, 5, 15, 30], "Compare k = 1, 5, 15 and 30."
    _rows = []
    for _k in k_values:
        _model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=_k))
        _model.fit(X_train, y_train)
        _rows.append({"k": _k, "training": _model.score(X_train, y_train), "validation": _model.score(X_val, y_val)})
    knn_results = pd.DataFrame(_rows)
    knn_results
    return (knn_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Fit and read a single tree (8 minutes)

    A decision tree asks a sequence of questions such as “petal length ≤ 2.45 cm?”.
    A **leaf** is a final prediction. `max_depth` limits how many splits a path can contain.

    1. Set `tree_depth` to **2**.
    2. Fit the provided tree on `X_train` and `y_train`.
    3. Read the resulting diagram. Which feature does the first split use? Follow a flower
    with petal length 1.4 cm through the tree. Which species does it predict?

    For each node, **samples** is the number of training examples that reach it, **value**
    counts the three classes in the order setosa/versicolor/virginica, and **gini** measures
    class mixture (0 means pure). The left branch satisfies `≤`; the right branch does not.
    Trees do not need standard scaling for these threshold splits.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_train, X_val, iris, mo, plot_tree, plt, y_train, y_val):
    tree_depth = 2
    tree_model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
    tree_fitted = tree_model.fit(X_train, y_train)
    mo.stop(tree_fitted is None, mo.md("✏️ Use tree_model.fit with the training data."))
    assert tree_depth == 2, "Use depth 2 so the first tree stays easy to read."
    assert tree_model.get_depth() <= 2
    _fig, _ax = plt.subplots(figsize=(11, 5))
    plot_tree(tree_model, feature_names=iris.feature_names, class_names=list(iris.target_names),
              filled=True, rounded=True, ax=_ax, fontsize=9)
    _fig.tight_layout()
    mo.vstack([mo.md(f"Tree validation accuracy: **{tree_model.score(X_val, y_val):.1%}**"), _fig])
    return (tree_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Does a deeper tree help? (5 minutes)

    Set `depth_values` to **[1, 2, 3, 5, None]**. Here `None` means there is no explicit maximum depth.
    The provided loop fits each model and reports its scores and number of leaves.

    Does validation accuracy improve whenever training accuracy improves? Compare the rows.
    A training improvement without a validation improvement can warn of overfitting, but
    deeper trees do not always perform worse. This small dataset and split may show no such case.
    Compare this with the k=1 result from Exercise 3.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_train, X_val, mo, pd, y_train, y_val):
    depth_values = [1, 2, 3, 5, None]
    mo.stop(depth_values is None, mo.md("✏️ Enter [1, 2, 3, 5, None]."))
    assert depth_values == [1, 2, 3, 5, None]
    _rows = []
    for _depth in depth_values:
        _model = DecisionTreeClassifier(max_depth=_depth, random_state=42).fit(X_train, y_train)
        _rows.append({"depth": "unlimited" if _depth is None else str(_depth),
                      "leaves": _model.get_n_leaves(), "training": _model.score(X_train, y_train),
                      "validation": _model.score(X_val, y_val)})
    tree_results = pd.DataFrame(_rows)
    tree_results
    return (tree_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Final comparison on unseen flowers (6 minutes)

    Use the **validation** tables to choose each model's setting. The provided code selects the
    highest validation accuracy, choosing the first listed setting if there is a tie.
    Then it refits using training + validation data.

    Fill in the two `predict` calls using `X_test`. The code below shows test accuracy,
    precision/recall/F1, and confusion matrices. Rows are actual species; columns are predictions.

    Which species get confused? Does either model win on this split? With only 30 test flowers,
    one extra mistake changes accuracy by about 3.3 percentage points. **Do not retune using these test scores.**
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, DecisionTreeClassifier, KNeighborsClassifier, StandardScaler, X_test, X_trainval, accuracy_score, classification_report, depth_values, iris, knn_results, make_pipeline, mo, np, pd, plt, tree_results, y_test, y_trainval):
    best_k = int(knn_results.loc[knn_results["validation"].idxmax(), "k"])
    best_depth = depth_values[int(tree_results["validation"].idxmax())]
    final_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_trainval, y_trainval)
    final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42).fit(X_trainval, y_trainval)
    knn_test_predictions = final_knn.predict(X_test)
    tree_test_predictions = final_tree.predict(X_test)
    mo.stop(knn_test_predictions is None or tree_test_predictions is None,
            mo.md("✏️ Predict X_test with final_knn and final_tree."))
    assert np.array_equal(knn_test_predictions, final_knn.predict(X_test))
    assert np.array_equal(tree_test_predictions, final_tree.predict(X_test))
    test_results = pd.DataFrame({"model": [f"KNN (k={best_k})", f"Tree (depth={best_depth})"],
                                 "test accuracy": [accuracy_score(y_test, knn_test_predictions), accuracy_score(y_test, tree_test_predictions)]})
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _ax, _name, _pred in zip(_axes, ["KNN", "Single tree"], [knn_test_predictions, tree_test_predictions]):
        ConfusionMatrixDisplay.from_predictions(y_test, _pred, labels=[0, 1, 2], display_labels=iris.target_names,
                                               colorbar=False, ax=_ax, cmap="Blues")
        _ax.set_title(_name)
    _fig.tight_layout()
    _knn_report = pd.DataFrame(classification_report(y_test, knn_test_predictions, target_names=iris.target_names, output_dict=True, zero_division=0)).T
    _tree_report = pd.DataFrame(classification_report(y_test, tree_test_predictions, target_names=iris.target_names, output_dict=True, zero_division=0)).T
    mo.vstack([test_results, _fig, mo.md("**KNN classification report**"), _knn_report,
               mo.md("**Tree classification report**"), _tree_report])
    return best_depth, best_k, final_knn, final_tree, knn_test_predictions, test_results, tree_test_predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Before you finish: explain what you saw

    Write a sentence for each question in the next cell:

    1. Why do we scale features for KNN?
    2. What does a perfect training score fail to tell us?
    3. Why did we keep validation and test data separate?
    """)
    return


@app.cell
def _(mo):
    reflection = """
    1. KNN uses distances, so a feature with much larger numeric units can dominate them.
       Standard scaling makes the feature scales comparable; it does not guarantee higher accuracy.
    2. A perfect training score only describes seen examples. It does not establish generalization.
    3. Validation chooses model settings; the held-out test estimates performance after those choices.
    """
    mo.md(reflection)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension A · A change of units

    The same measurement can be written in centimetres or micrometres. Below we multiply
    sepal length by 10,000, which changes its units but adds no new information.

    Run the cell and compare raw and scaled KNN on validation data. Why can changing a unit
    affect unscaled KNN? Scaling is not guaranteed to improve accuracy on every small dataset;
    the key point is that distance should not depend arbitrarily on the chosen units.
    """)
    return


@app.cell
def _(KNeighborsClassifier, StandardScaler, X_train, X_val, make_pipeline, pd, y_train, y_val):
    _rows = []
    for _factor in [1, 10_000]:
        _train = X_train.copy()
        _val = X_val.copy()
        _train[:, 0] *= _factor
        _val[:, 0] *= _factor
        for _name, _model in [
            ("Raw KNN", KNeighborsClassifier(n_neighbors=5)),
            ("Scaled KNN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ]:
            _model.fit(_train, y_train)
            _rows.append({"sepal length multiplier": _factor, "model": _name,
                          "validation accuracy": _model.score(_val, y_val)})
    pd.DataFrame(_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension B · Count the work

    For **one** new flower, brute-force KNN computes a distance to each of `n` training flowers.
    Each distance uses `m` feature coordinates. This distance-computation step is **O(n × m)**;
    selecting the neighbours adds work too. Efficient search structures can change prediction costs.

    If `n = 1,000` and `m = 4`, how many feature-coordinate comparisons contribute to all distances?
    Set `coordinate_comparisons` to the answer. What happens if you double `n`?

    For comparison, a tree prediction follows one path, with one feature test per visited split.
    For a path of depth `d`, that is O(d); an unbalanced tree can be much deeper than a balanced one.
    """)
    return


@app.cell
def _(mo):
    coordinate_comparisons = 1000 * 4
    mo.stop(coordinate_comparisons is None, mo.md("✏️ Multiply training examples by features."))
    assert coordinate_comparisons == 4000, "Each of 1,000 distances uses four features."
    mo.md("✅ 4,000 coordinate comparisons; doubling the training set gives 8,000. This counts coordinates, not exact CPU operations.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Instructor discussion notes

    - Exercise 1: versicolor wins 2–1. A neighbour vote is not a certainty estimate.
    - Exercises 2–3: compare validation scores, not just the training score. A tie is possible;
      use the stated rule for reproducibility rather than claiming a universal best k.
    - Exercise 4: read the feature and threshold printed at the root. The 1.4 cm petal takes
      the setosa branch for this fitted tree. Depth counts splits along a path, not total nodes.
    - Exercise 5: additional leaves can fit training detail without improving validation accuracy.
    - Exercise 6: discuss actual off-diagonal counts. Neither model is universally superior.
      With this small test set, avoid strong conclusions from one or two flowers.
    - Extension A: changing units alters raw distances; training-based standardization removes
      that positive unit multiplier (up to floating-point effects).
    - Extension B: brute-force distance computation scales with both training rows and features;
      the tree instead follows a path whose depth depends on how it was grown.

    References: [KNN](https://scikit-learn.org/stable/modules/neighbors.html),
    [decision trees](https://scikit-learn.org/stable/modules/tree.html),
    [data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
    """)
    return


if __name__ == "__main__":
    app.run()
