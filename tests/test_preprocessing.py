import nltk

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

from finsent.preprocessing.negation import get_negation_safe_stopwords
from finsent.preprocessing.text_cleaning import clean_for_lexicon_ml


def test_negation_tokens_survive_stopword_filtering():
    sw = get_negation_safe_stopwords()
    assert "not" not in sw
    assert "n't" not in sw
    assert "never" not in sw
    assert "the" in sw
    assert "is" in sw


def test_clean_for_lexicon_ml_preserves_negation():
    tokens = clean_for_lexicon_ml("The company is not profitable this quarter.")
    assert "not" in tokens
    assert "the" not in tokens
    assert "is" not in tokens


def test_clean_for_lexicon_ml_lemmatizes():
    tokens = clean_for_lexicon_ml("The companies reported strong earnings.")
    assert "company" in tokens
    assert "companies" not in tokens
