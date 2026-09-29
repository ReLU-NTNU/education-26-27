# Bootcamp 6 · Logistic regression and Naive Bayes examples

These are the existing Jupyter demonstrations from the 2025–2026 course,
imported unchanged as a starting point for this year's material.

- [Logistic regression](Logistic%20Regression%20Example.ipynb): breast-cancer
  classification, train/test splitting, scaling, classification metrics and ROC/AUC.
- [Gaussian Naive Bayes](naive%20bayes%20example.ipynb): Iris classification,
  class priors, feature means/variances and predicted probabilities.

These are worked examples, not yet student exercises or marimo notebooks.
They do not cover KNN or single decision trees.

## Status and known issues

This import preserves the original notebook code and saved outputs; it has not
been validated by running every cell. Before teaching from these examples:

- Correct the logistic-regression plot title: the dataset uses **0 = malignant,
  1 = benign**. The current ROC curve treats benign as the positive class.
- Define `beta_x_samples` and `y_pred_prob` before the final logistic-curve cell.
- Replace `gnb.sigma_` with `gnb.var_` in the Naive Bayes notebook. The original
  saved output contains an AttributeError for the old attribute.

These notebooks do not yet have a dedicated setup or launcher in this repo.
Do not install these notebooks' dependencies into the Bootcamp 2 environment.
Bootcamp 2 continues to use its existing scripts, lockfile and personal
`bootcamp_2/work/` notebook.

## Source

Copied from [tutorials-25-26 / notebooks_2025 / bootcamp_6](https://github.com/ReLU-NTNU/tutorials-25-26/tree/2a5a9fdb48549b3a620b7ba3e8bfac6f3d20f5f2/notebooks_2025/bootcamp_6)
at commit `2a5a9fdb48549b3a620b7ba3e8bfac6f3d20f5f2`.
