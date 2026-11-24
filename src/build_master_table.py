"""
Build the master table for Subtask 2: one row per sentence (usage).

It combines four SemEval files from data/SV/ into one CSV:
    usages.jsonl       the sentence, where the word is, year, period
    definitions.jsonl  the target definition for each word
    subtask2.jsonl     the gold 0/1 label  (what we predict)
    subtask1.jsonl     the gold sense ids  (only for error analysis!)

Output: data/interim/sv_master.csv

How to run:
    python src/build_master_table.py
"""

import json
from pathlib import Path

import pandas as pd

# File locations
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "SV"
OUTPUT_FILE = PROJECT_DIR / "data" / "interim" / "sv_master.csv"


def read_jsonl(path):
    """Read a .jsonl file (one JSON object per line) into a DataFrame."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() != "":
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    # ------------------------------------------------------------------
    # 1. Read the four files
    # ------------------------------------------------------------------
    usages = read_jsonl(DATA_DIR / "usages.jsonl")
    definitions = read_jsonl(DATA_DIR / "definitions.jsonl")
    subtask2 = read_jsonl(DATA_DIR / "subtask2.jsonl")
    subtask1 = read_jsonl(DATA_DIR / "subtask1.jsonl")

    # ------------------------------------------------------------------
    # 2. Keep only the columns we need, with clear names
    # ------------------------------------------------------------------
    labels = subtask2[["word", "sentence_id", "label"]]

    senses = subtask1[["word", "sentence_id", "label"]]
    senses = senses.rename(columns={"label": "gold_senses"})
    # The senses are lists like [3, 4]. Turn them into text like "3;4"
    # so they save nicely in a CSV.
    senses["gold_senses"] = senses["gold_senses"].apply(
        lambda sense_list: ";".join(str(s) for s in sense_list)
    )

    defs = definitions[["word", "definition"]].copy()
    # The 'motiv' definition has invisible soft hyphens (­); remove them
    defs["definition"] = defs["definition"].str.replace("­", "")

    # ------------------------------------------------------------------
    # 3. Join everything together
    #    Each sentence is identified by (word, sentence_id).
    #    Definitions are per word, so they join on word only.
    # ------------------------------------------------------------------
    df = usages.merge(labels, on=["word", "sentence_id"])
    df = df.merge(senses, on=["word", "sentence_id"])
    df = df.merge(defs, on="word")

    # ------------------------------------------------------------------
    # 4. Fix types and add a few helper columns
    # ------------------------------------------------------------------
    df["year"] = df["year"].astype(int)    # was text, e.g. "1883"
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["label"] = df["label"].astype(int)

    # target_form = the word exactly as it appears in the sentence
    target_forms = []
    for i in range(len(df)):
        text = df.loc[i, "text"]
        start = df.loc[i, "start"]
        end = df.loc[i, "end"]
        target_forms.append(text[start:end])
    df["target_form"] = target_forms

    # span_ok = a quick check that start/end really point at the word:
    # no spaces around it, and same first letter as the dictionary word.
    # (A few 'skär' rows fail this: their offsets point at the wrong word.)
    span_ok = []
    for i in range(len(df)):
        form = df.loc[i, "target_form"]
        word = df.loc[i, "word"]
        ok = form != "" and form == form.strip() and form[0].lower() == word[0].lower()
        span_ok.append(ok)
    df["span_ok"] = span_ok

    # era = "old" (1880-1915) or "modern" (2006-2023); nothing in between
    df["era"] = "modern"
    df.loc[df["year"] < 1950, "era"] = "old"

    # ------------------------------------------------------------------
    # 5. Put the columns in a nice order and save
    # ------------------------------------------------------------------
    columns = [
        "word", "sentence_id",
        "text", "start", "end", "target_form", "span_ok",
        "pos", "period_label", "year", "era",
        "definition",
        "label", "gold_senses",
    ]
    df = df[columns]

    # Simple checks
    if len(df) != len(usages):
        print("Warning: some sentences were lost in the join!")
    print("Sentences:", len(df))
    print("Words:", df["word"].nunique())
    print("Rows with suspicious spans:", (~df["span_ok"]).sum())
    print()
    print("Share of label 1 per word:")
    print(df.groupby("word")["label"].mean().round(2).to_string())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print()
    print("Saved", OUTPUT_FILE)


if __name__ == "__main__":
    main()
