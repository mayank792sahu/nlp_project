"""Loughran-McDonald finance-specific lexicon scorer: counts positive vs. negative
word hits in the preprocessed token list; majority wins, ties (including 0-0) -> neutral."""

from typing import Sequence

from finsent.lexicons.lm_lexicon import build_lm_lexicon
from finsent.models.base import SentimentModel
from finsent.preprocessing.text_cleaning import clean_for_lexicon_ml


class LoughranMcDonaldModel(SentimentModel):
    name = "Loughran-McDonald"

    def __init__(self):
        lexicon = build_lm_lexicon()
        self._positive_words = lexicon["positive"]
        self._negative_words = lexicon["negative"]

    def _score_tokens(self, tokens: list) -> tuple:
        pos = sum(1 for t in tokens if t in self._positive_words)
        neg = sum(1 for t in tokens if t in self._negative_words)
        return pos, neg

    def _label_for_counts(self, pos: int, neg: int) -> str:
        if pos == neg:
            return "neutral"
        return "positive" if pos > neg else "negative"

    def predict(self, texts: Sequence[str]) -> list:
        preds = []
        for text in texts:
            tokens = clean_for_lexicon_ml(text)
            pos, neg = self._score_tokens(tokens)
            preds.append(self._label_for_counts(pos, neg))
        return preds

    def predict_proba(self, texts: Sequence[str]) -> list:
        """Pseudo-scores from normalized positive/negative hit counts (not a calibrated
        probability, but useful to visualize how strongly the lexicon leaned each way)."""
        results = []
        for text in texts:
            tokens = clean_for_lexicon_ml(text)
            pos, neg = self._score_tokens(tokens)
            total = pos + neg
            if total == 0:
                results.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0})
            else:
                results.append({"positive": pos / total, "negative": neg / total, "neutral": 0.0})
        return results
