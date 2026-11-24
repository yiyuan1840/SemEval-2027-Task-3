"""
Build the feature table for the 9 training words.

One row per sentence of every training word. The test word (kvinna) is
left out completely: it has no rows here.

Columns added to the master table:
    tfidf_sim    cosine similarity between the sentence and its definition
                 (TF-IDF fit on that word's own sentences + definition, no labels)
    llm_score    log P("Ja") - log P("Nej") from llm_scores.py
    tfidf_sim_z  tfidf_sim as a z-score within its word
    llm_score_z  llm_score as a z-score within its word

The z-score columns put every word on the same scale (mean 0, std 1 within
each word). Both are computed without labels, so the same thing can be
done for a test word at prediction time.

Input:  data/interim/sv_master.csv, data/interim/sv_splits.csv
        data/processed/sv_llm_scores.csv   (made by llm_scores.py)
Output: data/processed/sv_train_features.csv

How to run (from the project folder):
    python src/build_train_features.py
"""

from pathlib import Path

import pandas as pd

from make_splits import load_split
from tfidf_models import make_vectorizer, definition_similarity

PROJECT_DIR = Path(__file__).resolve().parent.parent
LLM_FILE = PROJECT_DIR / "data" / "processed" / "sv_llm_scores.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "sv_train_features.csv"


def add_tfidf_sim(train):
    """Add a tfidf_sim column, computed one word at a time."""
    train["tfidf_sim"] = 0.0
    for word in train["word"].unique():
        rows = train["word"] == word
        word_df = train[rows]

        vectorizer = make_vectorizer()
        vectorizer.fit(list(word_df["text"]) + list(word_df["definition"].unique()))

        similarities = definition_similarity(vectorizer, word_df)
        train.loc[rows, "tfidf_sim"] = [s[0] for s in similarities]
    return train


def add_z_score(train, column):
    """Add <column>_z: the column as a z-score within each word."""
    word_mean = train.groupby("word")[column].transform("mean")
    word_std = train.groupby("word")[column].transform("std")
    train[column + "_z"] = (train[column] - word_mean) / word_std
    return train


def main():
    if not LLM_FILE.exists():
        print("Error:", LLM_FILE, "does not exist. Run src/llm_scores.py first.")
        raise SystemExit(1)

    # Only the training words; the test word is not in here
    train, test = load_split()
    print("Training words:", sorted(train["word"].unique()))
    print("Left out (test):", sorted(test["word"].unique()))

    # Add the LLM score for each sentence
    llm = pd.read_csv(LLM_FILE)
    train = train.merge(llm[["word", "sentence_id", "llm_score"]],
                        on=["word", "sentence_id"], how="left")
    missing = train["llm_score"].isna().sum()
    if missing > 0:
        print("Warning:", missing, "sentences have no LLM score")

    # Add the TF-IDF similarity
    train = add_tfidf_sim(train)

    # Within-word z-scores
    train = add_z_score(train, "tfidf_sim")
    train = add_z_score(train, "llm_score")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT_FILE, index=False)

    print("Rows:", len(train), " Words:", train["word"].nunique())
    print("Saved", OUTPUT_FILE)


if __name__ == "__main__":
    main()
