"""
Two trivial baselines: always guess 0, and always guess 1.

Any real model must beat these. Because the metric is macro-F1 (both
classes count equally), guessing one class for everything scores at most
about 0.5 per word, and close to 0 when that class is rare for the word.

Both baselines give the same ROC curve (the diagonal, AUC 0.5), so only
always_1 goes on the comparison plot. When you add a real model, run it
with run_cv and add its result to the list passed to plot_roc.

How to run (from the project folder):
    python src/baselines.py
"""

from evaluate_cv import run_cv, plot_roc


def always_0(train_part, val_part):
    return [0] * len(val_part)


def always_1(train_part, val_part):
    return [1] * len(val_part)


if __name__ == "__main__":
    result_0 = run_cv(always_0, "always_0")
    result_1 = run_cv(always_1, "always_1")

    plot_roc([result_1])
