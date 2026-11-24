"""
Leave-one-word-out cross-validation on the 9 training words.

A "model" here is just a function:

    def my_model(train_part, val_part):
        ...
        return scores

It gets the training sentences and the validation sentences, and returns
one score per validation sentence: a number between 0 and 1, where higher
means "more likely the target sense". A score >= 0.5 counts as label 1.

For each validation word we compute macro-F1, the same metric as the
official SemEval scorer. The final score is the average over the 9 words.

plot_roc() draws one ROC plot with a curve for every model, so they can be
compared. Each curve uses the scores from all 9 validation words together.

How to use it (see baselines.py for a full example):
    from evaluate_cv import run_cv, plot_roc
    result = run_cv(my_model, "my_model")
    plot_roc([result, other_result])
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from make_splits import load_split, get_cv_folds

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def run_cv(model, name):
    """
    Run leave-one-word-out CV for one model and print the scores.

    Returns a dict with the model name, the mean macro-F1, and all true
    labels and scores (needed for the ROC plot).
    """
    train, test = load_split()  # we never use test here

    f1_scores = []
    all_labels = []
    all_scores = []

    print("Model:", name)
    for val_word, train_part, val_part in get_cv_folds(train):
        true_labels = list(val_part["label"])
        scores = list(model(train_part, val_part))
        predictions = [1 if s >= 0.5 else 0 for s in scores]

        f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
        auc = roc_auc_score(true_labels, scores)
        print(f"  {val_word:10s}  macro-F1 = {f1:.3f}   AUC = {auc:.3f}")

        f1_scores.append(f1)
        all_labels = all_labels + true_labels
        all_scores = all_scores + scores

    mean_f1 = sum(f1_scores) / len(f1_scores)
    print(f"  {'MEAN':10s}  macro-F1 = {mean_f1:.3f}")
    print()

    return {"name": name, "mean_f1": mean_f1, "labels": all_labels, "scores": all_scores}


def plot_roc(results, filename="roc_comparison.png"):
    """Draw one ROC plot with a curve for each model in results."""
    plt.figure(figsize=(6, 6))

    for result in results:
        false_pos_rate, true_pos_rate, _ = roc_curve(result["labels"], result["scores"])
        auc = roc_auc_score(result["labels"], result["scores"])
        label = f"{result['name']} (AUC {auc:.2f}, F1 {result['mean_f1']:.2f})"
        plt.plot(false_pos_rate, true_pos_rate, label=label)

    plt.plot([0, 1], [0, 1], "k:", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC, all CV validation words")
    plt.legend(fontsize=8, loc="lower right")

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_file = RESULTS_DIR / filename
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print("Saved", plot_file)
