"""Parses the Loughran-McDonald Master Dictionary into positive/negative word sets.

The dictionary is not pip-installable; download it manually from the University of
Notre Dame SRAF site (https://sraf.nd.edu/loughranmcdonald-master-dictionary/) and
place the CSV (or XLSX) at data/external/LM_MasterDictionary.csv per config.yaml.

In the dictionary, a word belongs to a sentiment category if that category's column
holds a nonzero value (historically the year the word was added; some releases use a
plain 0/1 flag instead) rather than always being literally 0/1, so membership is
checked as `!= 0` rather than truthiness of a boolean.
"""

import pickle
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer

from finsent.config import CONFIG, resolve_path


def _load_raw_dictionary(csv_path: Path) -> pd.DataFrame:
    if csv_path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(csv_path)
    return pd.read_csv(csv_path)


def _lemmatize_set(words: set) -> set:
    lemmatizer = WordNetLemmatizer()
    return {lemmatizer.lemmatize(w) for w in words}


def build_lm_lexicon(force_reload: bool = False) -> dict:
    """Returns {'positive': set[str], 'negative': set[str]} of lemmatized, lowercased words."""
    lex_cfg = CONFIG["lexicons"]
    cache_path = resolve_path(lex_cfg["lm_cache_pkl"])

    if cache_path.exists() and not force_reload:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    csv_path = resolve_path(lex_cfg["loughran_mcdonald_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Loughran-McDonald dictionary not found at {csv_path}. Download it manually "
            "from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ and place it there."
        )

    df = _load_raw_dictionary(csv_path)
    df.columns = [c.strip() for c in df.columns]

    positive_words = set(df.loc[df["Positive"] != 0, "Word"].str.lower())
    negative_words = set(df.loc[df["Negative"] != 0, "Word"].str.lower())

    lexicon = {
        "positive": _lemmatize_set(positive_words),
        "negative": _lemmatize_set(negative_words),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(lexicon, f)

    return lexicon
