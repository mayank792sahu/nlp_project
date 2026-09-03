"""Negation-preserving stopword list.

Standard NLTK English stopwords minus a curated set of negation cues, so that
"not", "n't", "never", etc. survive stopword filtering and remain available to
downstream lexicon/ML scoring (which otherwise loses sentiment-flipping context).
"""

from nltk.corpus import stopwords

NEGATION_TOKENS = {
    "no", "not", "nor", "n't", "never", "none", "nobody", "nothing",
    "neither", "nowhere", "cannot", "without", "hardly", "scarcely", "barely",
}


def get_negation_safe_stopwords() -> set:
    return set(stopwords.words("english")) - NEGATION_TOKENS
