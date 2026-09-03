"""Tokenization, negation-safe stopword removal, and lemmatization.

Two entry points:
- `clean_for_lexicon_ml`: full pipeline (tokenize -> stopword filter -> lemmatize),
  used by the Loughran-McDonald scorer and the TF-IDF+ML model.
- VADER and FinBERT deliberately do NOT use this — they consume raw sentences,
  since both are built to interpret natural language directly.
"""

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from finsent.preprocessing.negation import get_negation_safe_stopwords

_lemmatizer = WordNetLemmatizer()
_stopwords = None


def _stopwords_set() -> set:
    global _stopwords
    if _stopwords is None:
        _stopwords = get_negation_safe_stopwords()
    return _stopwords


def tokenize(text: str) -> list:
    return word_tokenize(text.lower())


def remove_stopwords(tokens: list) -> list:
    sw = _stopwords_set()
    return [t for t in tokens if t not in sw]


def lemmatize(tokens: list) -> list:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def clean_for_lexicon_ml(text: str) -> list:
    """Returns the fully preprocessed token list: tokenize -> negation-safe stopword removal -> lemmatize."""
    tokens = tokenize(text)
    tokens = [t for t in tokens if t.isalpha() or t == "n't"]
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens


def clean_for_lexicon_ml_str(text: str) -> str:
    """Same as clean_for_lexicon_ml but joined back into a string, for TfidfVectorizer input."""
    return " ".join(clean_for_lexicon_ml(text))
