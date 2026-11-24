"""
Split the Swedish data into 1 test word and 9 training words.

Why split by word?
    The real SemEval test set uses different words from the dev set.
    So we hold out a whole word (all of its sentences) as our test set.
    A word is never split: all of its sentences go to train, or all go to test.

Cross-validation (CV) on the 9 training words:
    We use "leave-one-word-out". Each training word gets its own fold number
    (0 to 8). In each CV round, one word is the validation set and the other
    8 words are used for training.

Input:  data/interim/sv_master.csv   (made by build_master_table.py)
Output: data/interim/sv_splits.csv   (one row per word: word, split, fold)

How to run:
    python src/make_splits.py

To change the test word, edit TEST_WORD below.

How to use the split in another script (run from the src/ folder):
    from make_splits import load_split, get_cv_folds

    train, test = load_split()
    for val_word, train_part, val_part in get_cv_folds(train):
        ...  # fit on train_part, evaluate on val_part
"""

from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# The word we hold out as our final test set.
# kvinna has the most sentences (290), appears in both old and modern
# texts, and has a medium-low positive rate (23%).
TEST_WORD = "kvinna"

# File locations (built from the location of this script, so it works
# no matter which folder you run it from)
PROJECT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = PROJECT_DIR / "data" / "interim" / "sv_master.csv"
SPLITS_FILE = PROJECT_DIR / "data" / "interim" / "sv_splits.csv"


# ----------------------------------------------------------------------
# Step 1: decide which word goes where
# ----------------------------------------------------------------------

def make_splits(master):
    """Return a small table with one row per word: word, split, fold."""

    # All the words in the data, in alphabetical order
    all_words = sorted(master["word"].unique())

    if TEST_WORD not in all_words:
        print("Error: the test word", TEST_WORD, "is not in the data.")
        print("Choose one of:", all_words)
        raise SystemExit(1)

    rows = []

    # The test word gets fold -1 (it is never used in CV)
    rows.append({"word": TEST_WORD, "split": "test", "fold": -1})

    # Every other word is a training word with its own fold number
    fold_number = 0
    for word in all_words:
        if word == TEST_WORD:
            continue
        rows.append({"word": word, "split": "train", "fold": fold_number})
        fold_number = fold_number + 1

    splits = pd.DataFrame(rows)

    # Add some info about each word so the file is easy to read:
    #   n        = number of sentences
    #   pos_rate = share of sentences with label 1
    splits["n"] = 0
    splits["pos_rate"] = 0.0
    for i in range(len(splits)):
        word = splits.loc[i, "word"]
        word_rows = master[master["word"] == word]
        splits.loc[i, "n"] = len(word_rows)
        splits.loc[i, "pos_rate"] = round(word_rows["label"].mean(), 3)

    return splits


# ----------------------------------------------------------------------
# Step 2: helpers that other scripts can import
# ----------------------------------------------------------------------

def load_split():
    """
    Read the master table and the splits file.
    Return two DataFrames of sentences: (train, test).
    Each sentence gets a 'split' and a 'fold' column from its word.
    """
    # dtype=str keeps gold_senses like "3;4" as text
    master = pd.read_csv(MASTER_FILE, dtype={"gold_senses": str})
    splits = pd.read_csv(SPLITS_FILE)

    # Give every sentence the split and fold of its word
    data = master.merge(splits[["word", "split", "fold"]], on="word")

    train = data[data["split"] == "train"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    return train, test


def get_cv_folds(train):
    """
    Leave-one-word-out CV on the training words.

    Returns a list of (val_word, train_part, val_part):
      val_word   = the word held out in this round
      train_part = sentences of the other 8 words
      val_part   = sentences of val_word
    """
    folds = []
    for val_word in sorted(train["word"].unique()):
        train_part = train[train["word"] != val_word]
        val_part = train[train["word"] == val_word]
        folds.append((val_word, train_part, val_part))
    return folds


# ----------------------------------------------------------------------
# Main: make the splits, save them, and print a summary
# ----------------------------------------------------------------------

def main():
    master = pd.read_csv(MASTER_FILE, dtype={"gold_senses": str})

    splits = make_splits(master)
    splits.to_csv(SPLITS_FILE, index=False)

    print(splits.to_string(index=False))
    print()

    train, test = load_split()
    print("Train:", train["word"].nunique(), "words,", len(train), "sentences")
    print("Test: ", TEST_WORD, ",", len(test), "sentences")

    # Safety check: no word should be in both train and test
    shared_words = set(train["word"]) & set(test["word"])
    if len(shared_words) > 0:
        print("Error: these words are in both train and test:", shared_words)
        raise SystemExit(1)

    print()
    print("CV folds (leave-one-word-out):")
    for val_word, train_part, val_part in get_cv_folds(train):
        print("  validate on", val_word, "-", len(train_part), "train /", len(val_part), "val sentences")

    print()
    print("Saved", SPLITS_FILE)


if __name__ == "__main__":
    main()
