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
def import_marimo():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # Bootcamp 6 · Neighbours and decision trees

    **Worked solutions.** Try the exercise notebook before reading these answers.

    By the end, you will be able to make a neighbour vote, implement KNN, fit KNN and a single
    decision tree, choose model settings using validation data, and read a confusion matrix.

    **How to work:** replace `None` at each `WRITE HERE` comment, then press
    **Shift+Enter**. Unfinished cells show a hint and dependent cells wait. Save with
    **Ctrl+S / Cmd+S**. Each cell uses descriptive names; use a new name for a new
    result. Work through Exercises 1–7; the last two are short extensions.
    In the notebook editor you edit the code inside each cell; marimo manages the
    outer cell functions in the saved Python file.

    No forest, boosting, GPU or dataset download is needed. The Iris data comes with scikit-learn.
    """)
    return


@app.cell
def import_libraries():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sklearn.datasets import load_iris
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    return (
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        StandardScaler,
        accuracy_score,
        classification_report,
        load_iris,
        make_pipeline,
        np,
        pd,
        plot_tree,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def data_instructions(mo):
    mo.md(r"""
    ### Meet the data (provided)

    Each row is a flower. Four **features** describe sepal and petal length/width in centimetres.
    The **target** is its species: 0 = setosa, 1 = versicolor, 2 = virginica.

    We split the 150 flowers into **90 training**, **30 validation**, and **30 test** examples.
    Training data fits the models. Validation data helps choose `k` and tree depth.
    Keep the test set for the final comparison in Exercise 7. `stratify` keeps class proportions
    balanced; `random_state` makes the split repeatable.
    """)
    return


@app.cell
def prepare_data(load_iris, mo, pd, train_test_split):
    # Load the built-in flower measurements and their known species labels.
    iris = load_iris()

    # X holds the four measurements for every flower; each row is one flower.
    X = iris.data

    # y holds the matching species label for each row of X (0, 1 or 2).
    y = iris.target

    # Reserve 20% (30 flowers) for the final test; do not use them to choose settings.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Take 25% of the remaining 120 flowers for validation, leaving 90 for training.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        stratify=y_trainval,
        random_state=42,
    )

    # Show six training flowers so we can see what one row of input looks like.
    data_preview = pd.DataFrame(X_train[:6], columns=iris.feature_names)

    # Add the matching species name beside each row of measurements.
    data_preview["species"] = iris.target_names[y_train[:6]]

    mo.vstack(
        [
            mo.md(
                f"**Split:** {len(y_train)} train / {len(y_val)} validation / {len(y_test)} test"
            ),
            data_preview,
        ]
    )
    return (
        X_test,
        X_train,
        X_trainval,
        X_val,
        iris,
        y_test,
        y_train,
        y_trainval,
        y_val,
    )


@app.cell(hide_code=True)
def vote_instructions(mo):
    mo.md(r"""
    ## 1 · Predict one flower by counting votes (3 minutes)

    **The situation:** we have a new flower and do not know its species. KNN looks
    at flowers whose species we already know. We have already found the **three
    closest flowers** for you, so you do not need to calculate any distances yet.

    Each neighbour gets **one vote for its own species**:

    | Neighbour | Known species | Its vote |
    |---|---|---|
    | Closest flower | versicolor | versicolor |
    | Second closest | virginica | virginica |
    | Third closest | versicolor | versicolor |

    **Your task:** predict the new flower's species by choosing the name that
    appears most often in the vote column. This is called a **majority vote**.
    All three votes count equally; do not just choose the first row.

    For example, votes `["cat", "dog", "dog"]` would predict `"dog"`, because
    dog has two votes and cat has one. Apply the same counting rule to the flowers.

    **What to type:** in the next cell, replace only `None` with your chosen
    flower name in quotes. `None` is a placeholder meaning “no answer yet”.
    Quotes tell Python that your answer is text (a **string**).
    For the animal example, the line would be `neighbour_vote = "dog"`;
    your answer must use one of the flower names from the table instead.

    Keep `neighbour_vote =` as it is. Press **Shift+Enter** to run the cell.
    You should get either a green confirmation or a hint to count again.
    In Exercise 2, you will write the code that finds and counts neighbours.
    """)
    return


@app.cell
def neighbour_vote_exercise(mo):
    # Store your predicted species as text, with quotation marks around the name.
    neighbour_vote = "versicolor"

    mo.stop(
        neighbour_vote is None,
        mo.md("✏️ Replace None with the class receiving the most votes."),
    )
    return (neighbour_vote,)


@app.cell(hide_code=True)
def check_neighbour_vote(mo, neighbour_vote):
    if neighbour_vote == "versicolor":
        vote_feedback = "✅ Correct! Versicolor has two votes and virginica has one. Predict **versicolor**."
    else:
        vote_feedback = "Count the vote column again: which flower name appears **twice**? Check your spelling too."
    mo.md(vote_feedback)
    return


@app.cell(hide_code=True)
def manual_knn_instructions(mo):
    mo.md(r"""
    ## 2 · Build KNN from the lecture (12 minutes)

    Before using a library, complete `knn_predict` below. It follows the lecture:
    **measure distances → find the nearest neighbours → look up labels → vote**.

    `X_train` contains known flowers, `y_train` their labels, and `X_new` the flowers
    to predict. The loop handles one new flower at a time; `predictions` collects
    one answer per flower. These function arguments and variables are ordinary
    Python names local to the function.

    Fill the four `None` blanks:

    1. **Distances:** `np.linalg.norm(X_train - flower, axis=1)` gives one Euclidean
       distance per training row. Subtraction compares every row with the new
       flower; `axis=1` combines the four feature differences in each row.
    2. **Neighbours:** `np.argsort(distances)` returns row indices from closest to
       furthest. Use `[:k]` to keep the first `k` indices.
    3. **Labels:** use those indices to select entries from `y_train`.
    4. **Vote:** `np.bincount(labels, minlength=3)` counts votes for labels 0, 1, 2.
       `np.argmax(vote_counts)` gives the label with the most votes.

    For example, labels `[1, 2, 1]` give counts `[0, 2, 1]`, so class **1** wins.
    The lecture's `sum(labels) > k / 2` vote works only for **binary labels 0 and 1**.
    Iris has three classes, so we count each class separately. If votes tie, we
    choose the smallest class label. Even an odd `k` can tie with three classes.

    Run the next cell to try your function. Explain what `[:k]` changes, and why
    summing the labels would give the wrong vote for `[0, 0, 2]`.
    """)
    return


@app.cell
def implement_knn(np):
    # Define a reusable prediction function; k=3 is the default if no k is supplied.
    def knn_predict(X_train, y_train, X_new, k=3):
        predictions = []

        # Handle each new flower in turn; the indented lines repeat for every row.
        for flower in X_new:
            # one distance for each training flower
            # Subtract this flower from every training row, then measure each row of
            # differences.
            distances = np.linalg.norm(X_train - flower, axis=1)

            # indices of the k closest training flowers
            # Sorting gives row positions; [:k] keeps only the k smallest-distance
            # positions.
            nearest = np.argsort(distances)[:k]

            # labels belonging to these neighbours
            # Use those row positions to look up the known species of the nearest
            # flowers.
            labels = y_train[nearest]

            # Count labels 0, 1 and 2 separately; the species numbers are names, not
            # quantities.
            vote_counts = np.bincount(labels, minlength=3)

            # the label with the largest vote count
            # Choose the position of the largest count; its position is the species
            # label.
            prediction = np.argmax(vote_counts)

            predictions.append(prediction)

        return np.array(predictions)

    return (knn_predict,)


@app.cell
def try_manual_knn(
    StandardScaler,
    X_train,
    X_val,
    accuracy_score,
    knn_predict,
    mo,
    pd,
    y_train,
    y_val,
):
    # Create an object that will learn the mean and spread of each feature.
    manual_scaler = StandardScaler()

    # Learn the scaling values from TRAINING data only to avoid leaking validation
    # information.
    manual_scaler.fit(X_train)

    # Rescale the training features using the means and spreads just learned.
    manual_train = manual_scaler.transform(X_train)

    # Use the SAME training scale for validation; do not learn a new one here.
    manual_validation = manual_scaler.transform(X_val)

    # Call your function for every validation flower, using three neighbours.
    manual_predictions = knn_predict(
        manual_train,
        y_train,
        manual_validation,
        k=3,
    )

    # Compare predictions with known validation labels; 1.0 means all are correct.
    manual_accuracy = accuracy_score(y_val, manual_predictions)

    manual_preview = pd.DataFrame({
        "actual label": y_val[:6],
        "predicted label": manual_predictions[:6],
    })

    mo.vstack([
        mo.md(f"Your KNN validation accuracy: **{manual_accuracy:.1%}**"),
        manual_preview,
    ])
    return


@app.cell(hide_code=True)
def knn_instructions(mo):
    mo.md(r"""
    ## 3 · Use scikit-learn for KNN (7 minutes)

    You have built the distance-and-vote steps yourself. Now use a library to do
    the same kind of work, with scaling and prediction kept together in a pipeline.

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
def fit_knn(
    KNeighborsClassifier,
    StandardScaler,
    X_train,
    X_val,
    accuracy_score,
    make_pipeline,
    mo,
    np,
    y_train,
    y_val,
):
    # k is how many nearby training flowers get to vote for each prediction.
    k_start = 5

    # Apply scaling first, then KNN; the pipeline repeats these steps on new data.
    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k_start),
    )

    # Fit the pipeline on training features X_train and their labels y_train.
    knn_fitted = knn_model.fit(X_train, y_train)

    # Predict the labels of X_val, which was not used to fit this model.
    knn_predictions = knn_model.predict(X_val)

    mo.stop(
        knn_predictions is None, mo.md("✏️ Call predict with the validation features.")
    )

    assert k_start == 5, "Use five neighbours for this starting model."

    assert np.asarray(knn_predictions).shape == y_val.shape, (
        "Return one prediction per validation flower."
    )

    assert np.array_equal(knn_predictions, knn_model.predict(X_val)), (
        "Use this model's validation predictions."
    )

    mo.md(
        f"✅ KNN validation accuracy: **{accuracy_score(y_val, knn_predictions):.1%}**"
    )
    return


@app.cell(hide_code=True)
def knn_comparison_instructions(mo):
    mo.md(r"""
    ## 4 · What changes when k changes? (6 minutes)

    Set `k_values` to **[1, 5, 15, 30]** and run the provided experiment.
    The table shows training and validation accuracy, not test accuracy.

    Compare the two columns. Does the model that best fits training data also do best on validation?
    A small `k` can follow very local details; a larger `k` averages over a wider neighbourhood.
    A perfect training score is not evidence of perfect performance on new flowers.
    """)
    return


@app.cell
def compare_neighbours(
    KNeighborsClassifier,
    StandardScaler,
    X_train,
    X_val,
    make_pipeline,
    mo,
    pd,
    y_train,
    y_val,
):
    # List the neighbour counts we want to compare, from very local to broader votes.
    k_values = [1, 5, 15, 30]

    mo.stop(k_values is None, mo.md("✏️ Enter the four k values as a list."))

    assert k_values == [1, 5, 15, 30], "Compare k = 1, 5, 15 and 30."

    knn_rows = []

    # Repeat the experiment for each k in the list.
    for neighbour_count in k_values:
        # Create a fresh scaled KNN model for this neighbour count.
        candidate_knn = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=neighbour_count),
        )

        # Fit using training features and labels; validation data is only used for
        # scoring.
        candidate_knn.fit(X_train, y_train)

        knn_rows.append(
            {
                "k": neighbour_count,
                "training": candidate_knn.score(X_train, y_train),
                "validation": candidate_knn.score(X_val, y_val),
            }
        )

    knn_results = pd.DataFrame(knn_rows)
    knn_results
    return (knn_results,)


@app.cell(hide_code=True)
def tree_instructions(mo):
    mo.md(r"""
    ## 5 · Fit and read a single tree (8 minutes)

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
def fit_tree(
    DecisionTreeClassifier,
    X_train,
    X_val,
    iris,
    mo,
    plot_tree,
    plt,
    y_train,
    y_val,
):
    # Limit the number of splits along any path from the root to a leaf.
    tree_depth = 2

    # Create a tree with that depth; the fixed seed makes this example repeatable.
    tree_model = DecisionTreeClassifier(
        max_depth=tree_depth,
        random_state=42,
    )

    # Let the tree learn its questions from training flowers and their labels.
    tree_fitted = tree_model.fit(X_train, y_train)

    mo.stop(tree_fitted is None, mo.md("✏️ Use tree_model.fit with the training data."))

    assert tree_depth == 2, "Use depth 2 so the first tree stays easy to read."

    assert tree_model.get_depth() <= 2

    tree_figure, tree_axis = plt.subplots(figsize=(11, 5))

    # Draw the learned splits and label the features and predicted species.
    plot_tree(
        tree_model,
        feature_names=iris.feature_names,
        class_names=list(iris.target_names),
        filled=True,
        rounded=True,
        ax=tree_axis,
        fontsize=9,
    )

    tree_figure.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"Tree validation accuracy: **{tree_model.score(X_val, y_val):.1%}**"
            ),
            tree_figure,
        ]
    )
    return


@app.cell(hide_code=True)
def depth_instructions(mo):
    mo.md(r"""
    ## 6 · Does a deeper tree help? (5 minutes)

    Set `depth_values` to **[1, 2, 3, 5, None]**. Here `None` means there is no explicit maximum depth.
    The provided loop fits each model and reports its scores and number of leaves.

    Does validation accuracy improve whenever training accuracy improves? Compare the rows.
    A training improvement without a validation improvement can warn of overfitting, but
    deeper trees do not always perform worse. This small dataset and split may show no such case.
    Compare this with the k=1 result from Exercise 4.
    """)
    return


@app.cell
def compare_depths(
    DecisionTreeClassifier,
    X_train,
    X_val,
    mo,
    pd,
    y_train,
    y_val,
):
    # Compare four depth limits and None, which means no explicit depth limit.
    depth_values = [1, 2, 3, 5, None]

    mo.stop(depth_values is None, mo.md("✏️ Enter [1, 2, 3, 5, None]."))

    assert depth_values == [1, 2, 3, 5, None]

    tree_rows = []

    # Repeat the experiment for each depth setting.
    for candidate_depth in depth_values:
        # Create a fresh tree for this depth before fitting it on training data.
        candidate_tree = DecisionTreeClassifier(
            max_depth=candidate_depth,
            random_state=42,
        )

        # Fit using training features and labels; validation data is only used for
        # scoring.
        candidate_tree.fit(X_train, y_train)

        if candidate_depth is None:
            depth_label = "unlimited"
        else:
            depth_label = str(candidate_depth)

        tree_rows.append(
            {
                "depth": depth_label,
                "leaves": candidate_tree.get_n_leaves(),
                "training": candidate_tree.score(X_train, y_train),
                "validation": candidate_tree.score(X_val, y_val),
            }
        )

    tree_results = pd.DataFrame(tree_rows)
    tree_results
    return depth_values, tree_results


@app.cell(hide_code=True)
def evaluation_instructions(mo):
    mo.md(r"""
    ## 7 · Final comparison on unseen flowers (6 minutes)

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
def evaluate_models(
    DecisionTreeClassifier,
    KNeighborsClassifier,
    StandardScaler,
    X_test,
    X_trainval,
    depth_values,
    knn_results,
    make_pipeline,
    mo,
    np,
    tree_results,
    y_trainval,
):
    # Find the row with the highest VALIDATION score; idxmax picks the first tie.
    best_knn_row = knn_results["validation"].idxmax()

    # Read the neighbour count from that row and convert it to a Python integer.
    best_k = int(knn_results.loc[best_knn_row, "k"])

    # Find the best tree row using validation accuracy, without looking at test data.
    best_tree_row = tree_results["validation"].idxmax()

    # Use that row position to get the original depth setting, including None.
    best_depth = depth_values[int(best_tree_row)]

    # Build KNN using the setting chosen on validation data.
    final_knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=best_k),
    )

    # Refit on training plus validation now that the settings are chosen; leave test
    # data untouched.
    final_knn.fit(X_trainval, y_trainval)

    # Build a tree using the depth chosen on validation data.
    final_tree = DecisionTreeClassifier(
        max_depth=best_depth,
        random_state=42,
    )

    # Refit on training plus validation now that the settings are chosen; leave test
    # data untouched.
    final_tree.fit(X_trainval, y_trainval)

    # Predict the held-out test flowers with the refitted KNN model.
    knn_test_predictions = final_knn.predict(X_test)

    # Predict the same test flowers with the refitted tree.
    tree_test_predictions = final_tree.predict(X_test)

    mo.stop(
        knn_test_predictions is None or tree_test_predictions is None,
        mo.md("✏️ Predict X_test with final_knn and final_tree."),
    )

    assert np.array_equal(knn_test_predictions, final_knn.predict(X_test))

    assert np.array_equal(tree_test_predictions, final_tree.predict(X_test))
    return best_depth, best_k, knn_test_predictions, tree_test_predictions


@app.cell(hide_code=True)
def display_evaluation(
    ConfusionMatrixDisplay,
    accuracy_score,
    best_depth,
    best_k,
    classification_report,
    iris,
    knn_test_predictions,
    mo,
    pd,
    plt,
    tree_test_predictions,
    y_test,
):
    test_results = pd.DataFrame(
        {
            "model": [f"KNN (k={best_k})", f"Tree (depth={best_depth})"],
            "test accuracy": [
                accuracy_score(y_test, knn_test_predictions),
                accuracy_score(y_test, tree_test_predictions),
            ],
        }
    )
    confusion_figure, confusion_axes = plt.subplots(1, 2, figsize=(10, 4))
    for confusion_axis, model_name, model_predictions in zip(
        confusion_axes, ["KNN", "Single tree"], [knn_test_predictions, tree_test_predictions]
    ):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            model_predictions,
            labels=[0, 1, 2],
            display_labels=iris.target_names,
            colorbar=False,
            ax=confusion_axis,
            cmap="Blues",
        )
        confusion_axis.set_title(model_name)
    confusion_figure.tight_layout()
    knn_report = pd.DataFrame(
        classification_report(
            y_test,
            knn_test_predictions,
            target_names=iris.target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    tree_report = pd.DataFrame(
        classification_report(
            y_test,
            tree_test_predictions,
            target_names=iris.target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    mo.vstack(
        [
            test_results,
            confusion_figure,
            mo.md("**KNN classification report**"),
            knn_report,
            mo.md("**Tree classification report**"),
            tree_report,
        ]
    )
    return


@app.cell(hide_code=True)
def reflection_instructions(mo):
    mo.md(r"""
    ### Before you finish: explain what you saw

    Write a sentence for each question in the next cell:

    1. Why do we scale features for KNN?
    2. What does a perfect training score fail to tell us?
    3. Why did we keep validation and test data separate?
    """)
    return


@app.cell
def reflect(mo):
    # Write your explanations inside these triple quotes, which allow several text
    # lines.
    reflection = """
    1. KNN uses distances, so a feature with much larger numeric units can dominate them.
       Standard scaling makes the feature scales comparable; it does not guarantee higher accuracy.
    2. A perfect training score only describes seen examples. It does not establish generalization.
    3. Validation chooses model settings; the held-out test estimates performance after those choices.
    """

    mo.md(reflection)
    return


@app.cell(hide_code=True)
def scaling_instructions(mo):
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
def compare_units(
    KNeighborsClassifier,
    StandardScaler,
    X_train,
    X_val,
    make_pipeline,
    pd,
    y_train,
    y_val,
):
    scaling_rows = []

    # Try unchanged units (1) and a 10,000-fold unit conversion.
    for unit_multiplier in [1, 10_000]:
        # Copy the training array so this experiment cannot change the original data.
        changed_train = X_train.copy()

        # Make a separate validation copy for the same unit change.
        changed_validation = X_val.copy()

        # [:, 0] selects every row of the first feature; multiply it to change its
        # units.
        changed_train[:, 0] *= unit_multiplier

        # [:, 0] selects every row of the first feature; multiply it to change its
        # units.
        changed_validation[:, 0] *= unit_multiplier

        for scaling_name, scaling_model in [
            ("Raw KNN", KNeighborsClassifier(n_neighbors=5)),
            (
                "Scaled KNN",
                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
            ),
        ]:
            # Fit using training features and labels; validation data is only used for
            # scoring.
            scaling_model.fit(changed_train, y_train)

            scaling_rows.append(
                {
                    "sepal length multiplier": unit_multiplier,
                    "model": scaling_name,
                    "validation accuracy": scaling_model.score(changed_validation, y_val),
                }
            )

    pd.DataFrame(scaling_rows)
    return


@app.cell(hide_code=True)
def complexity_instructions(mo):
    mo.md(r"""
    ## Extension B · Count the work

    For **one** new flower, brute-force KNN computes a distance to each of `n` training flowers.
    Each distance uses `m` feature coordinates. This distance-computation step is **O(n × m)**;
    our `np.argsort` also sorts all `n` distances, adding **O(n log n)** work per query.
    Efficient search structures can change prediction costs.

    If `n = 1,000` and `m = 4`, how many feature-coordinate comparisons contribute to all distances?
    Set `coordinate_comparisons` to the answer. What happens if you double `n`?

    For comparison, a tree prediction follows one path, with one feature test per visited split.
    For a path of depth `d`, that is O(d); an unbalanced tree can be much deeper than a balanced one.
    """)
    return


@app.cell
def count_comparisons(mo):
    # Each training flower uses four feature comparisons; multiply rows by features.
    coordinate_comparisons = 1000 * 4

    mo.stop(
        coordinate_comparisons is None,
        mo.md("✏️ Multiply training examples by features."),
    )

    assert coordinate_comparisons == 4000, "Each of 1,000 distances uses four features."

    mo.md(
        "✅ 4,000 coordinate comparisons; doubling the training set gives 8,000. This counts coordinates, not exact CPU operations."
    )
    return


@app.cell(hide_code=True)
def discussion_notes(mo):
    mo.md(r"""
    ## Instructor discussion notes

    - Exercise 1: versicolor wins 2–1. A neighbour vote is not a certainty estimate.
    - Exercise 2: count each class separately. `[0, 0, 2]` votes for 0, not 1.
      A three-way tie chooses the smallest label.
    - Exercises 3–4: compare validation scores, not just the training score. A tie is possible;
      use the stated rule for reproducibility rather than claiming a universal best k.
    - Exercise 5: read the feature and threshold printed at the root. The 1.4 cm petal takes
      the setosa branch for this fitted tree. Depth counts splits along a path, not total nodes.
    - Exercise 6: additional leaves can fit training detail without improving validation accuracy.
    - Exercise 7: discuss actual off-diagonal counts. Neither model is universally superior.
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
