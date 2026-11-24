"""
Plot how each word's two senses change over time (gold labels).

For each of the 9 validation (training) words:
    x axis: time period
    y axis: the two senses, as two rows
        "target sense"  = label 1 (the sense in the definition)
        "other senses"  = label 0
    each dot: the share of that period's sentences in that sense
              (bigger dot = bigger share, and the % is written on it)

The sense with the larger share in the most recent period is the top row.
The test word is left out on purpose, so we don't look at it.

Output:
    results/timeseries/<word>.png   one plot per word
    results/timeseries/all_words.png   all 9 words in one grid

How to run (from the project folder):
    python src/plot_sense_timeseries.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from make_splits import load_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "results" / "timeseries"

# All 10 periods in time order. There is a big gap between 1915 and 2006.
ALL_PERIODS = [
    "1880-1883", "1888-1891", "1896-1899", "1904-1907", "1912-1915",
    "2006-2007", "2010-2011", "2014-2015", "2018-2019", "2022-2023",
]

TARGET_COLOR = "#2a78d6"  # blue
OTHER_COLOR = "#eb6834"   # orange
TEXT_COLOR = "#333333"


def sense_shares(word_df):
    """
    For one word, return a list of (period, n, target_share) in time order,
    only for periods that have sentences.
    """
    rows = []
    for period in ALL_PERIODS:
        period_df = word_df[word_df["period_label"] == period]
        if len(period_df) > 0:
            target_share = period_df["label"].mean()
            rows.append((period, len(period_df), target_share))
    return rows


def plot_word(ax, word, word_df, short_labels=False):
    """
    Draw the two-row time plot for one word on the given axes.
    short_labels=True shows only the start year (e.g. "1880"), for the grid.
    """
    rows = sense_shares(word_df)

    # Which sense goes on top? The one with the larger share in the latest period.
    latest_target_share = rows[-1][2]
    target_on_top = latest_target_share >= 0.5
    if target_on_top:
        target_y, other_y = 1, 0
    else:
        target_y, other_y = 0, 1

    for x, (period, n, target_share) in enumerate(rows):
        other_share = 1 - target_share
        for y, share, color in [(target_y, target_share, TARGET_COLOR),
                                (other_y, other_share, OTHER_COLOR)]:
            # Dot area grows with the share; skip empty dots
            if share > 0:
                ax.scatter(x, y, s=1200 * share, color=color, alpha=0.85,
                           edgecolor="white", linewidth=2, zorder=3)
            ax.text(x, y - 0.32, f"{share:.0%}", ha="center", va="center",
                    fontsize=8, color=TEXT_COLOR)

    # Dashed line where the time gap between the two eras is
    periods_shown = [r[0] for r in rows]
    for x in range(1, len(periods_shown)):
        if periods_shown[x] == "2006-2007":
            ax.axvline(x - 0.5, color="#999999", linestyle="--", linewidth=1)

    # Axes and labels
    ax.set_yticks([target_y, other_y])
    ax.set_yticklabels(["target sense", "other senses"])
    ax.set_ylim(-0.6, 1.5)
    ax.set_xticks(range(len(rows)))
    tick_labels = []
    for period, n, _ in rows:
        if short_labels:
            period = period[:4]
        tick_labels.append(f"{period}\nn={n}")
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_title(word, fontsize=11)

    # Quiet frame: no box, no grid
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    train, test = load_split()  # only use the 9 validation words
    words = sorted(train["word"].unique())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One plot per word
    for word in words:
        word_df = train[train["word"] == word]
        fig, ax = plt.subplots(figsize=(8, 2.8))
        plot_word(ax, word, word_df)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{word}.png", dpi=150)
        plt.close(fig)

    # All 9 words in one 3 x 3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 9))
    for ax, word in zip(axes.flatten(), words):
        plot_word(ax, word, train[train["word"] == word], short_labels=True)
    fig.suptitle("Share of sentences in each sense, per period (gold labels)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_words.png", dpi=150)
    plt.close(fig)

    print("Saved", len(words), "word plots and all_words.png to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
