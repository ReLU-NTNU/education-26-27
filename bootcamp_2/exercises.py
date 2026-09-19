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
    # Bootcamp 2 · Python for machine learning

    Work through **NumPy → pandas → Matplotlib → PyTorch**. Fill the `WRITE HERE` blanks, then run the cell with **Shift+Enter**. Checks give feedback after you enter an answer.

    marimo reruns dependent cells when a value changes. Each variable must be defined in only one cell; names starting with `_` are local to that cell. Save with **Ctrl+S** (macOS: **Cmd+S**).

    Unfinished exercises show a reminder; dependent exercises wait for their inputs. The MNIST section has a button to load/download the data. No GPU is needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Setup
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    print(device)
    print("Environment ready: NumPy, pandas, Matplotlib and CPU PyTorch.")
    return DataLoader, datasets, nn, np, pd, plt, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. NumPy (arrays & operations)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.01 (create an array)**: Create a 1D numpy array `simple_array` with values `[1, 2, 3]`.
    """)
    return


@app.cell
def _(mo, np):
    # WRITE HERE: create a 1D array with values [1,2,3]
    simple_array = None

    mo.stop(any(_answer is None for _answer in (simple_array,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("simple_array:", simple_array)
    assert isinstance(simple_array, np.ndarray), "simple_array should be a numpy array"
    assert np.array_equal(simple_array, np.array([1, 2, 3])), "simple_array should equal [1,2,3]"
    print("OK 1.01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.02 (zeros/ones)**: Create an array of zeros `zeros_array` that looks like this:

    $$
    \begin{bmatrix}
    0.0 & 0.0 \\
    0.0 & 0.0
    \end{bmatrix}
    $$

    And a 1x3 ones array `ones_array` that looks like this:

    $$
    \begin{bmatrix}
    1.0 & 1.0 & 1.0
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _(mo, np):
    # WRITE HERE: create zeros and ones arrays with dtype float64
    zeros_array = None
    ones_array = None

    mo.stop(any(_answer is None for _answer in (zeros_array, ones_array,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("zeros_array:\n", zeros_array)
    print("ones_array:\n", ones_array)
    assert isinstance(zeros_array, np.ndarray) and isinstance(ones_array, np.ndarray), "zeros_array and ones_array should be arrays"
    assert zeros_array.shape == (2,2) and ones_array.shape == (1,3), "Unexpected shapes for zeros_array or ones_array"
    assert np.allclose(zeros_array, 0.0) and np.allclose(ones_array, 1.0), "Values should be all zeros / all ones"
    print("OK 1.02")
    assert zeros_array.dtype == np.float64 and ones_array.dtype == np.float64, "Use dtype float64"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.03 (dtype)**: Create an array with `dtype=np.float32`
    """)
    return


@app.cell
def _(mo, np):
    # WRITE HERE: create array with dtype float32
    float32_array = None

    mo.stop(any(_answer is None for _answer in (float32_array,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("float32_array dtype:", float32_array.dtype)
    assert isinstance(float32_array, np.ndarray), "float32_array should be a numpy array"
    assert float32_array.dtype == np.float32, "float32_array should have dtype float32"
    print("OK 1.03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.04 (reshape)**: Reshape `linear_range = np.arange(6)` into:

    $$
    \begin{bmatrix}
    0 & 1 & 2 \\
    3 & 4 & 5
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _(mo, np):
    linear_range = np.arange(6) # [0, 1, 2, 3, 4, 5]

    # WRITE HERE: reshape linear_range to (2,3)
    reshaped_to_2x3 = None

    mo.stop(any(_answer is None for _answer in (reshaped_to_2x3,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("reshaped_to_2x3:\n", reshaped_to_2x3)
    assert isinstance(reshaped_to_2x3, np.ndarray), "reshaped_to_2x3 should be a numpy array"
    assert reshaped_to_2x3.shape == (2,3), "reshaped_to_2x3 should have shape (2,3)"
    assert np.array_equal(reshaped_to_2x3, np.array([[0,1,2],[3,4,5]])), "reshaped_to_2x3 values unexpected"
    print("OK 1.04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.05 (elementwise ops)**: For `base_array = np.array([1,2,3])`, compute `plus_five = base_array + 5` and `times_two = base_array * 2`.
    """)
    return


@app.cell
def _(mo, np):
    base_array = np.array([1, 2, 3])

    # WRITE HERE: elementwise add and multiply
    plus_five = None
    times_two = None

    mo.stop(any(_answer is None for _answer in (plus_five, times_two,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("plus_five:", plus_five)
    print("times_two:", times_two)
    assert np.array_equal(plus_five, np.array([6, 7, 8])), "plus_five should be base_array+5"
    assert np.array_equal(times_two, np.array([2, 4, 6])), "times_two should be base_array*2"
    print("OK 1.05")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.06 (indexing)**: For `numbers = np.array([[2, 3, 4], [5, 6, 7]])`, store the third element in the second row (value 7) in scalar `third_value_second_row`.
    """)
    return


@app.cell
def _(mo, np):
    numbers = np.array([[2, 3, 4],
                        [5, 6, 7]])

    # WRITE HERE: index the third element into third_value (a scalar)
    third_value_second_row = None

    mo.stop(any(_answer is None for _answer in (third_value_second_row,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("third_value_second_row:", third_value_second_row)
    assert np.isscalar(third_value_second_row), "third_value_second_row should be a scalar"
    assert third_value_second_row == 7, "Expected value 7"
    print("OK 1.06")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.07 (slicing)**: For `values = np.array([10,20,30,40])`, select the first two elements into `first_two_elements` and the last two into `last_two_elements`.
    """)
    return


@app.cell
def _(mo, np):
    values = np.array([10, 20, 30, 40])

    # WRITE HERE: slice first two elements into first_two_elements
    first_two_elements = None

    # WRITE HERE: slice last two elements into last_two_elements
    last_two_elements = None

    mo.stop(any(_answer is None for _answer in (first_two_elements, last_two_elements,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("last_two_elements:", last_two_elements)
    assert isinstance(last_two_elements, np.ndarray), "last_two_elements should be a numpy array"
    assert last_two_elements.shape == (2,), "last_two_elements should have shape (2,)"
    assert np.array_equal(last_two_elements, np.array([30, 40])), "Expected [30, 40]"

    print("first_two_elements:", first_two_elements)
    assert isinstance(first_two_elements, np.ndarray), "first_two_elements should be a numpy array"
    assert first_two_elements.shape == (2,), "first_two_elements should have shape (2,)"
    assert np.array_equal(first_two_elements, np.array([10, 20])), "Expected [10, 20]"

    print("OK 1.07")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.08 (broadcasting add)**: Given `ones_matrix_2x3`
    $$
    \begin{bmatrix}
    1.0 & 1.0 & 1.0 \\
    1.0 & 1.0 & 1.0
    \end{bmatrix}
    $$

    and `row_vector_1x3 = np.array([1,2,3])`,
    $$
    \begin{bmatrix}
    1.0 & 2.0 & 3.0
    \end{bmatrix}
    $$

    sum them together so that you get the broadcasted result `broadcast_sum`:
    $$
    \begin{bmatrix}
    2.0 & 3.0 & 4.0 \\
    2.0 & 3.0 & 4.0
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _(mo, np):
    ones_matrix_2x3 = np.ones((2,3))
    row_vector_1x3  = np.array([1, 2, 3])

    # WRITE HERE: compute broadcast_sum using broadcasting
    broadcast_sum = None

    mo.stop(any(_answer is None for _answer in (broadcast_sum,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("broadcast_sum:\n", broadcast_sum)
    assert isinstance(broadcast_sum, np.ndarray), "broadcast_sum should be a numpy array"
    assert broadcast_sum.shape == (2,3), "broadcast_sum should have shape (2,3)"
    expected = np.vstack([row_vector_1x3 + 1, row_vector_1x3 + 1])
    assert np.array_equal(broadcast_sum, expected), f"Expected two rows of {row_vector_1x3}"
    print("OK 1.08")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.09 (boolean mask)**: From `array_1_to_5 = np.array([1,2,3,4,5])`, select values greater than 3 into `greater_than_three` so that you get:

    $$
    \begin{bmatrix}
    4 & 5
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _(mo, np):
    array_1_to_5 = np.array([1, 2, 3, 4, 5])

    # WRITE HERE: use a boolean mask to select values > 3
    greater_than_three = None

    mo.stop(any(_answer is None for _answer in (greater_than_three,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("greater_than_three:", greater_than_three)
    assert isinstance(greater_than_three, np.ndarray), "greater_than_three should be a numpy array"
    assert np.array_equal(greater_than_three, np.array([4, 5])), "Expected [4, 5]"
    print("OK 1.09")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.10 (view vs copy)**: Given `base_array_0_to_3 = np.arange(4)`:

    $$
    \begin{bmatrix}
    0 & 1 & 2 & 3
    \end{bmatrix}
    $$

     create a view `every_second_view = base_array_0_to_3[::2]` and a copy `every_second_copy = base_array_0_to_3[::2].copy()`. Then mutate `base_array_0_to_3[0] = 99` and observe the difference.
    """)
    return


@app.cell
def _(mo, np):
    base_array_0_to_3 = np.arange(4)

    # WRITE HERE: create a view and a copy from base_array_0_to_3 using slicing
    every_second_view = None
    every_second_copy = None

    # Mutate base
    base_array_0_to_3[0] = 99

    mo.stop(any(_answer is None for _answer in (every_second_view, every_second_copy,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("every_second_view:", every_second_view)
    print("every_second_copy:", every_second_copy)
    assert isinstance(every_second_view, np.ndarray) and isinstance(every_second_copy, np.ndarray), "every_second_view and every_second_copy should be arrays"
    assert np.array_equal(every_second_view, base_array_0_to_3[::2]), "every_second_view should reflect changes in base_array_0_to_3"
    assert np.array_equal(every_second_copy, np.array([0, 2])), "every_second_copy should be independent (values before mutation)"
    assert every_second_view[0] == 99 and every_second_copy[0] == 0, "view sees mutation, copy does not"
    print("OK 1.10")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.11 (sum)**: For `matrix_2x3 = np.array([[1,2,3],[4,5,6]])`:

    Compute the total sum into `total_sum` so that you get the scalar value: `21`.

    And the row-wise sum into `row_sums` so that you get:

    $$
    \begin{bmatrix}
    6 & 15
    \end{bmatrix}
    $$

    And the column-wise sum into `column_sums` so that you get:

    $$
    \begin{bmatrix}
    5 & 7 & 9
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _(mo, np):
    matrix_2x3 = np.array([[1, 2, 3],
                             [4, 5, 6]])

    # WRITE HERE: compute the sum along axis=1 into row_sums
    row_sums = None
    column_sums = None

    total_sum = None

    mo.stop(any(_answer is None for _answer in (row_sums, column_sums, total_sum,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("row_sums:", row_sums)
    print("column_sums:", column_sums)
    print("total_sum:", total_sum)
    assert isinstance(row_sums, np.ndarray), "row_sums should be a numpy array"
    assert row_sums.shape == (2,), "row_sums should have shape (2,)"
    assert np.array_equal(row_sums, np.array([6, 15])), "Expected [6, 15]"
    assert isinstance(column_sums, np.ndarray), "column_sums should be a numpy array"
    assert column_sums.shape == (3,), "column_sums should have shape (3,)"
    assert np.array_equal(column_sums, np.array([5, 7, 9])), "Expected [5, 7, 9]"
    assert np.isscalar(total_sum), "total_sum should be a scalar"
    assert total_sum == 21, "Expected value 21"
    print("OK 1.11")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Pandas (mini-ETL)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.01 (Series)**: Create a Pandas Series `scores_series` from the list `[10, 15, 12]`.
    """)
    return


@app.cell
def _(mo, pd):
    # WRITE HERE: create a Series from a list
    scores_series = None

    mo.stop(any(_answer is None for _answer in (scores_series,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("scores_series:\n", scores_series)
    assert isinstance(scores_series, pd.Series), "scores_series should be a Series"
    assert scores_series.tolist() == [10,15,12], "scores_series should contain [10,15,12]"
    print("OK 2.01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.02 (DataFrame from dict)**: Create `df_basic = DataFrame({name:["Alice","Bob","Charlie"], score:[10,15,12]})`.
    """)
    return


@app.cell
def _(mo, pd):
    # WRITE HERE: create a DataFrame
    df_basic = None

    mo.stop(any(_answer is None for _answer in (df_basic,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print(df_basic)
    assert isinstance(df_basic, pd.DataFrame), "df_basic should be a DataFrame"
    assert list(df_basic.columns) == ["name","score"], "Columns should be ['name','score']"
    assert len(df_basic) == 3, "df_basic should have 3 rows"
    print("OK 2.02")
    return (df_basic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.03 (select column)**: From `df_basic`, select the `score` column as a Series into `score_col` with values `[10,15,12]`.
    """)
    return


@app.cell
def _(mo, pd):
    # WRITE HERE: select score column as Series
    score_col = None

    mo.stop(any(_answer is None for _answer in (score_col,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("score_col:", score_col.values)
    assert isinstance(score_col, pd.Series), "score_col should be a Series"
    assert score_col.tolist() == [10,15,12], "score_col values should be [10,15,12]"
    print("OK 2.03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.04 (row filter)**: From `df_basic`, select rows with `score >= 12` into `df_sel` so that it contains names `Bob` and `Charlie` only.
    """)
    return


@app.cell
def _(mo, pd):
    # WRITE HERE: filter rows where score >= 12
    df_sel = None

    mo.stop(any(_answer is None for _answer in (df_sel,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print(df_sel)
    assert isinstance(df_sel, pd.DataFrame), "df_sel should be a DataFrame"
    assert list(df_sel["name"]) == ["Bob","Charlie"], "df_sel should contain names B and C"
    print("OK 2.04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.05 (assign new column)**: From `df_basic`, create `df_with_passed` by adding a boolean column `passed = score >= 12`.
    """)
    return


@app.cell
def _(df_basic, mo, pd):
    df_with_passed = df_basic.copy()
    # WRITE HERE: add a boolean column 'passed'
    None

    mo.stop("passed" not in df_with_passed.columns, mo.md("✏️ Add the **passed** column above, then run this cell."))

    # CHECK
    print(df_with_passed)
    assert isinstance(df_with_passed, pd.DataFrame), "df_with_passed should be a DataFrame"
    assert "passed" in df_with_passed.columns and df_with_passed["passed"].dtype == bool, "'passed' column should exist and be boolean"
    assert df_with_passed["passed"].tolist() == [False, True, True], "Expected passed values [False, True, True]"
    print("OK 2.05")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.06 (groupby mean)**: Compute the mean score per `passed` value into `mean_by_passed` so that you get the averages:

    False: 10.0

    True: 13.5
    """)
    return


@app.cell
def _(mo, np, pd):
    # WRITE HERE: compute mean score per 'passed'
    mean_by_passed = None

    mo.stop(any(_answer is None for _answer in (mean_by_passed,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("mean_by_passed:\n", mean_by_passed)
    assert isinstance(mean_by_passed, pd.Series), "mean_by_passed should be a Series"
    assert set(mean_by_passed.index.tolist()) == {False, True}, "Index should be booleans {False, True}"
    assert np.isclose(mean_by_passed.loc[False], 10.0) and np.isclose(mean_by_passed.loc[True], 13.5), "Expected means {False:10.0, True:13.5}"
    print("OK 2.06")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.07 (top-k)**: Select the top 2 rows by `score` into `top2`.
    """)
    return


@app.cell
def _(mo, pd):
    # WRITE HERE: select top 2 by score into top2
    top2 = None

    mo.stop(any(_answer is None for _answer in (top2,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print(top2)
    assert isinstance(top2, pd.DataFrame), "top2 should be a DataFrame"
    assert list(top2["name"]) == ["Bob","Charlie"], "Top 2 by score should be Bob and Charlie"
    print("OK 2.07")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.08 (missing values)**: Given a `score` column with one `NaN`, fill it with the column mean into `filled_scores` (a Series) so there are no missing values.
    """)
    return


@app.cell
def _(mo, np, pd):
    df_na = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "score": [10, np.nan, 12]})

    # WRITE HERE: fill NaN in 'score' with the column mean
    filled_scores = None

    mo.stop(any(_answer is None for _answer in (filled_scores,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("filled_scores:", filled_scores.values)
    assert isinstance(filled_scores, pd.Series), "filled_scores should be a Series"
    assert not filled_scores.isna().any(), "filled_scores should have no NaNs"
    expected_mean = np.mean([10, 12])
    assert np.isclose(filled_scores.iloc[1], expected_mean), "Filled value should equal column mean"
    print("OK 2.08")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.09 (merge/join)**: Merge two tables on `name`. Left-join `df_join_left` with `df_info` to produce `df_merged` that contains columns `name, score, age`.
    """)
    return


@app.cell
def _(mo, pd):
    df_join_left = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "score": [10,15,12]})
    df_info  = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [20,21,22]})

    # WRITE HERE: left-join on 'name'
    df_merged = None

    mo.stop(any(_answer is None for _answer in (df_merged,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print(df_merged)
    assert isinstance(df_merged, pd.DataFrame), "df_merged should be a DataFrame"
    assert list(df_merged.columns) == ["name","score","age"], "Columns should be ['name','score','age']"
    assert df_merged.loc[df_merged["name"]=="Bob","age"].iloc[0] == 21, "Age for Bob should be 21"
    print("OK 2.09")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.10 (value_counts)**: Using a DataFrame with a boolean column `passed`, compute the count of True/False values into `passed_counts` so that you get:

    True: 2

    False: 1
    """)
    return


@app.cell
def _(mo, pd):
    # Start from a small DataFrame
    pd_df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "score": [10,15,12]})
    df_counts_input = pd_df.assign(passed=lambda d: d["score"] >= 12)

    # WRITE HERE: count True/False in 'passed' into passed_counts
    passed_counts = None

    mo.stop(any(_answer is None for _answer in (passed_counts,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    print("passed_counts:\n", passed_counts)
    assert isinstance(passed_counts, pd.Series), "passed_counts should be a Series"
    assert set(passed_counts.index.tolist()) == {False, True}, "Index should be booleans {False, True}"
    assert passed_counts.loc[True] == 2 and passed_counts.loc[False] == 1, "Counts should be True:2, False:1"
    print("OK 2.10")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Matplotlib (basic plot)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.1**: For `line_x = np.arange(0,10)`, compute `line_y = line_x**2` and store in variable `line_y`. Then make a basic line plot with plt.plot (no special styling).
    """)
    return


@app.cell
def _(mo, np, plt):
    line_x = np.arange(0,10)

    # WRITE HERE
    line_y = None
    _fig, _ax = plt.subplots()
    # WRITE HERE: basic line plot of line_x vs line_y


    mo.stop(any(_answer is None for _answer in (line_y,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    mo.stop(not _ax.lines, mo.md("✏️ Add the plot above, then run this cell."))

    # CHECK
    assert isinstance(line_y, np.ndarray), "line_y should be a numpy array"
    assert line_y.shape == line_x.shape, "line_y should match line_x shape"
    assert np.all(line_y == line_x**2), "line_y should be line_x**2"
    print("OK 3.1")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.2 (scatter)**: Create a scatter plot of `(scatter_x, scatter_y)` where both arrays are filled with 10 random values between 0 and 1.
    """)
    return


@app.cell
def _(mo, np, plt):
    # DATA
    scatter_x = np.random.rand(10)
    scatter_y = np.random.rand(10)

    _fig, _ax = plt.subplots()
    # WRITE HERE: create a scatter plot of scatter_x vs scatter_y


    mo.stop(not _ax.collections, mo.md("✏️ Add the plot above, then run this cell."))

    # CHECKS
    assert isinstance(scatter_x, np.ndarray) and isinstance(scatter_y, np.ndarray), "scatter_x and scatter_y should be numpy arrays"
    assert scatter_x.shape == scatter_y.shape == (10,), "scatter_x and scatter_y should both have shape (10,)"
    print("OK 3.2 (scatter)")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.3 (MNIST image)**: Load one MNIST sample from the training set and display it with `plt.imshow(..., cmap='gray')` and `plt.axis('off')`.
    """)
    return


@app.cell
def _(mo):
    load_mnist = mo.ui.run_button(label="Load / download MNIST")
    load_mnist
    return (load_mnist,)


@app.cell
def _(datasets, load_mnist, mo, transforms):
    mo.stop(not load_mnist.value, mo.md("Click **Load / download MNIST** when you reach this section. The first load needs internet; data is stored in your OS cache, outside this repository."))
    from platformdirs import user_cache_path

    data_dir = user_cache_path("relu-education", appauthor=False) / "datasets"
    tx = transforms.ToTensor()
    mnist_train = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tx)
    mnist_test = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tx)
    mo.md(f"MNIST ready: **{len(mnist_train):,}** training images and **{len(mnist_test):,}** test images.")
    return mnist_test, mnist_train


@app.cell
def _(mnist_train, mo, plt):
    img_tensor, label = mnist_train[0]

    image = img_tensor.squeeze(0)

    _fig, _ax = plt.subplots()
    # WRITE HERE: display image as grayscale, no axes


    mo.stop(not _ax.images, mo.md("✏️ Add the plot above, then run this cell."))

    # CHECKS (shape only)
    assert img_tensor.shape == (1, 28, 28), "MNIST image should be 1x28x28"
    assert isinstance(label, int), "label should be an int"
    print("OK 3.3 (MNIST image)")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. PyTorch (data & model)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.1**: Create MNIST train/val/test loaders as `train_dl`, `val_dl`, `test_dl` using the given transform and splits (5k train, 1k val). Make sure to set batch size to 64 and shuffle the training data.
    """)
    return


@app.cell
def _(DataLoader, mnist_test, mnist_train, mo, torch):
    # Available inputs: mnist_train, mnist_test, random_split, DataLoader.
    mo.stop(mnist_train is None or mnist_test is None)
    # WRITE HERE: split mnist_train with random_split into 5000 training examples,
    # then split the remainder to get 1000 validation examples.
    train_ds = None
    remainder = None
    val_ds = None

    train_dl = None
    val_dl   = None
    test_dl  = None

    mo.stop(any(_answer is None for _answer in (train_ds, remainder, val_ds, train_dl, val_dl, test_dl,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK
    # Split sizes
    assert len(train_ds) == 5000 and len(val_ds) == 1000, "Incorrect split sizes (expected 5000 train, 1000 val)"

    # Loader types and batch sizes
    assert isinstance(train_dl, DataLoader) and isinstance(val_dl, DataLoader) and isinstance(test_dl, DataLoader), "loaders must be DataLoader instances"
    assert train_dl.batch_size == 64 and val_dl.batch_size == 64 and test_dl.batch_size == 64, "All loaders should use batch size 64"

    # Sample a batch and validate shapes/dtypes
    xb, yb = next(iter(train_dl))
    assert xb.ndim == 4 and xb.shape[1:] == (1, 28, 28), "MNIST images should be [B,1,28,28]"
    assert yb.ndim == 1 and xb.shape[0] == yb.shape[0], "Labels should be [B] and match batch size"
    assert xb.dtype == torch.float32, "Images should be float32 (after ToTensor)"
    assert yb.dtype == torch.int64, "Labels should be int64 class indices 0..9"
    assert yb.min().item() >= 0 and yb.max().item() <= 9, "Labels must be in [0,9]"

    print("OK 4.1")
    return train_dl, val_dl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.2**: Define a minimal 2-layer MLP named `model`: 28*28 → 128 → 10 with ReLU, using `nn.Sequential` and `nn.Flatten`. Move it to CPU.
    """)
    return


@app.cell
def _(mo, nn):
    # WRITE HERE
    model = None

    mo.stop(any(_answer is None for _answer in (model,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK: parameter count in a plausible range
    def count_params(m): return sum(p.numel() for p in m.parameters())
    assert isinstance(model, nn.Sequential), "Use nn.Sequential for this exercise"
    n_params = count_params(model)
    assert 100_000 <= n_params <= 110_000, f"Param count seems off: {n_params}"
    print("OK 4.2 (params ~", n_params, ")")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.3**: Implement a tiny training loop for 2 epochs with SGD (lr=0.1) and CrossEntropyLoss. Track `val_acc` after each epoch and store the final value in `final_val_acc`. Keep code minimal.
    """)
    return


@app.cell
def _(mo, model, train_dl, val_dl):
    # Use train_dl, val_dl and model from exercises 4.1 and 4.2.
    mo.stop(any(_value is None for _value in (train_dl, val_dl, model)))
    # Hints: use model.train() / model.eval(), opt.zero_grad(), loss.backward(), opt.step()
    # WRITE HERE
    final_val_acc = None

    mo.stop(any(_answer is None for _answer in (final_val_acc,)), mo.md("✏️ Fill in the **WRITE HERE** blanks above, then run this cell."))

    # CHECK (sanity only): final_val_acc should be between 0 and 1
    assert isinstance(final_val_acc, float), "final_val_acc should be a float"
    assert 0.0 <= final_val_acc <= 1.0, "final_val_acc must be within [0,1]"
    print("OK 4.3 (ran)")
    return


if __name__ == "__main__":
    app.run()
