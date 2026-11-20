"""
Build the master table for Subtask 2: one row per usage.

Joins the raw SemEval files for one language into a single table that every
system (logistic regression, neural net, ...) reads from:

    data/<LANG>/usages.jsonl       text, span, pos, period, year
    data/<LANG>/definitions.jsonl  one target definition per word
    data/<LANG>/subtask2.jsonl     gold 0/1 label
    data/<LANG>/subtask1.jsonl     gold sense ids (kept for error analysis)

Output: data/interim/<lang>_master.csv

The raw files are never modified. No text cleaning happens here (lowercasing,
masking the target word, etc.) because that differs per system.

Usage:
    python src/build_master_table.py            # defaults to SV
    python src/build_master_table.py --lang SV
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"

KEY = ["word", "sentence_id"]


def read_jsonl(path):
    """Read a .jsonl file into a DataFrame, one JSON object per line."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def check_unique_keys(df, name):
    n_dupes = df.duplicated(subset=KEY).sum()
    assert n_dupes == 0, f"{name}: {n_dupes} duplicate (word, sentence_id) keys"


def span_looks_ok(row):
    """
    Heuristic check that text[start:end] is actually the target word.

    A valid span has no leading/trailing whitespace and starts with the same
    letter as the lemma (this allows inflections like 'kvinnor' and irregular
    forms like 'sög' for 'suga'). It catches the few SV 'skär' rows whose
    offsets point at the wrong token (' hon', 'Hela ', ...).
    """
    form = row["target_form"]
    if form == "" or form != form.strip():
        return False
    return form[0].lower() == row["word"][0].lower()


def build(lang):
    lang_dir = DATA_DIR / lang

    usages = read_jsonl(lang_dir / "usages.jsonl")
    definitions = read_jsonl(lang_dir / "definitions.jsonl")
    subtask2 = read_jsonl(lang_dir / "subtask2.jsonl")
    subtask1 = read_jsonl(lang_dir / "subtask1.jsonl")

    # --- checks on the raw files ---------------------------------------
    for df, name in [(usages, "usages"), (subtask2, "subtask2"), (subtask1, "subtask1")]:
        check_unique_keys(df, name)

    usage_keys = set(zip(usages["word"], usages["sentence_id"]))
    label_keys = set(zip(subtask2["word"], subtask2["sentence_id"]))
    sense_keys = set(zip(subtask1["word"], subtask1["sentence_id"]))
    assert usage_keys == label_keys, "usages and subtask2 have different keys"
    assert usage_keys == sense_keys, "usages and subtask1 have different keys"

    assert definitions["word"].is_unique, "a word has more than one definition"
    missing_defs = set(usages["word"]) - set(definitions["word"])
    assert not missing_defs, f"words with no definition: {missing_defs}"

    # --- join -----------------------------------------------------------
    labels = subtask2[KEY + ["label"]]

    senses = subtask1[KEY + ["label"]].rename(columns={"label": "gold_senses"})
    # store the sense list as a string like "3;4" so it survives a CSV round trip
    senses["gold_senses"] = senses["gold_senses"].apply(
        lambda ids: ";".join(str(i) for i in ids)
    )

    defs = definitions[["word", "definition"]].copy()
    # the SV 'motiv' definition contains soft hyphens (U+00AD) that split tokens
    defs["definition"] = defs["definition"].str.replace("­", "", regex=False)

    df = usages.merge(labels, on=KEY, how="left", validate="one_to_one")
    df = df.merge(senses, on=KEY, how="left", validate="one_to_one")
    df = df.merge(defs, on="word", how="left", validate="many_to_one")

    # --- derived columns ------------------------------------------------
    df["year"] = df["year"].astype(int)  # raw year is a string, e.g. "1883"
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    df["target_form"] = [t[s:e] for t, s, e in zip(df["text"], df["start"], df["end"])]
    df["span_ok"] = df.apply(span_looks_ok, axis=1)

    # periods are either 1880-1915 or 2006-2023, nothing in between
    df["era"] = df["year"].apply(lambda y: "old" if y < 1950 else "modern")

    df["label"] = df["label"].astype(int)

    columns = [
        "word", "sentence_id",
        "text", "start", "end", "target_form", "span_ok",
        "pos", "period_label", "year", "era",
        "definition",
        "label", "gold_senses",
    ]
    df = df[columns]

    # --- checks on the result -------------------------------------------
    assert len(df) == len(usages), "row count changed during the join"
    assert df[["definition", "label", "gold_senses"]].notna().all().all(), "missing values after join"
    assert set(df["label"].unique()) <= {0, 1}, "labels other than 0/1"

    return df


def report(df):
    print(f"rows: {len(df)}   words: {df['word'].nunique()}")

    print("\nlabel counts per word (0 / 1, positive rate):")
    counts = df.groupby("word")["label"].agg(n="size", pos="sum")
    counts["neg"] = counts["n"] - counts["pos"]
    counts["pos_rate"] = (counts["pos"] / counts["n"]).round(2)
    print(counts[["neg", "pos", "n", "pos_rate"]].to_string())

    print("\nusages per word and era:")
    print(pd.crosstab(df["word"], df["era"]).to_string())

    bad = df[~df["span_ok"]]
    print(f"\nrows with suspicious spans: {len(bad)}")
    if len(bad):
        print(bad[["word", "sentence_id", "target_form"]].to_string(index=False))

    dupes = df[df.duplicated(subset=["word", "text", "start", "end"], keep=False)]
    if len(dupes):
        print(f"\nexact duplicate usages (same text and span, different id): {len(dupes)} rows")
        groups = dupes.groupby(["word", "text", "start", "end"])["label"].nunique()
        n_conflicts = (groups > 1).sum()
        print(f"  duplicate groups with conflicting labels: {n_conflicts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default="SV", help="language folder under data/ (default: SV)")
    args = parser.parse_args()

    df = build(args.lang)
    report(df)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / f"{args.lang.lower()}_master.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
