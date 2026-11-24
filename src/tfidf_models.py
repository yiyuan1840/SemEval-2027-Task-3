"""
TF-IDF models, evaluated with leave-one-word-out CV.

Two ways to use TF-IDF:

1. tfidf_sentence_lr
   Turn each sentence into a TF-IDF vector and train logistic regression
   on those vectors. This learns which words in a sentence go with label 1.
   Problem: the cue words are different for every target word (words around
   "herre" tell you nothing about "suga"), so this should NOT work well on
   an unseen word. It is here as a comparison.

2. tfidf_definition_sim
   Turn the sentence AND its definition into TF-IDF vectors and use their
   cosine similarity as the score. "Does this sentence look like the
   definition?" works for any word. No labels are used: nothing is trained.

For model 1, the TF-IDF vectorizer is fit inside each CV fold on the
training words only, so nothing from the validation word leaks into it.
Model 2 uses no labels, so it fits TF-IDF on the validation word's own
sentences and definition.

We use character n-grams (pieces of 3-5 letters) instead of whole words.
That way "kvinnor" and "kvinna", or old spellings like "öfver" and "över",
still share most of their pieces.

How to run (from the project folder):
    python src/tfidf_models.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

from baselines import always_1
from evaluate_cv import run_cv, plot_roc


def make_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)


def tfidf_sentence_lr(train_part, val_part):
    """Model 1: logistic regression on the sentence's TF-IDF vector."""
    model = make_pipeline(
        make_vectorizer(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    model.fit(train_part["text"], train_part["label"])
    return model.predict_proba(val_part["text"])[:, 1]


def definition_similarity(vectorizer, df):
    """For each row: cosine similarity between its sentence and its definition."""
    sentence_vectors = vectorizer.transform(df["text"])
    definition_vectors = vectorizer.transform(df["definition"])

    similarities = []
    for i in range(len(df)):
        sim = cosine_similarity(sentence_vectors[i], definition_vectors[i])[0, 0]
        similarities.append([sim])  # a list of one feature per row
    return similarities


def tfidf_definition_sim(train_part, val_part):
    """
    Model 2: no labels at all. The score is just how similar each sentence
    is to its definition.

    The vectorizer is fit on the validation word's own sentences plus its
    definition. That uses no labels, so it is allowed: at test time we also
    get all of a word's sentences at once. train_part is not used.
    """
    vectorizer = make_vectorizer()
    vectorizer.fit(list(val_part["text"]) + list(val_part["definition"].unique()))

    similarities = [row[0] for row in definition_similarity(vectorizer, val_part)]

    # Raw cosine similarities are small (often below 0.2), so a 0.5 cutoff
    # would give all 0s. Rescale within the word so the least similar
    # sentence is 0 and the most similar is 1; then >= 0.5 means "closer to
    # the most similar sentence than to the least similar one".
    lowest = min(similarities)
    highest = max(similarities)
    return [(s - lowest) / (highest - lowest) for s in similarities]


if __name__ == "__main__":
    result_baseline = run_cv(always_1, "always_1")
    result_sentence = run_cv(tfidf_sentence_lr, "tfidf_sentence_lr")
    result_similarity = run_cv(tfidf_definition_sim, "tfidf_definition_sim")

    plot_roc([result_baseline, result_sentence, result_similarity])
