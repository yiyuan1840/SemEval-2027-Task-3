"""
Run the whole data pipeline in one go.

    data/SV/*.jsonl  (raw SemEval files)
        |
        |  step 1: build_master_table.py
        v
    data/interim/sv_master.csv   (one row per sentence)
        |
        |  step 2: make_splits.py
        v
    data/interim/sv_splits.csv   (one row per word: test word + 9 CV folds)

How to run (from the project folder):
    python src/run_pipeline.py

To change the test word, edit TEST_WORD in make_splits.py.
"""

import build_master_table
import make_splits


def main():
    print("=" * 60)
    print("Step 1: building the master table from the raw .jsonl files")
    print("=" * 60)
    build_master_table.main()

    print()
    print("=" * 60)
    print("Step 2: splitting into 1 test word and 9 training words")
    print("=" * 60)
    make_splits.main()

    print()
    print("Done. Both files are in data/interim/")


if __name__ == "__main__":
    main()
