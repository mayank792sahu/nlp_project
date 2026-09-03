"""VADER: general-purpose lexicon/rule-based sentiment, used as-is on raw sentences."""

from typing import Sequence

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from finsent.config import CONFIG
from finsent.models.base import SentimentModel


class VaderModel(SentimentModel):
    name = "VADER"

    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()
        cfg = CONFIG["vader"]
        self._pos_threshold = cfg["pos_threshold"]
        self._neg_threshold = cfg["neg_threshold"]

    def _label_for_compound(self, compound: float) -> str:
        if compound >= self._pos_threshold:
            return "positive"
        if compound <= self._neg_threshold:
            return "negative"
        return "neutral"

    def predict(self, texts: Sequence[str]) -> list:
        return [self._label_for_compound(self._analyzer.polarity_scores(t)["compound"]) for t in texts]

    def predict_proba(self, texts: Sequence[str]) -> list:
        results = []
        for t in texts:
            scores = self._analyzer.polarity_scores(t)
            results.append({"positive": scores["pos"], "negative": scores["neg"], "neutral": scores["neu"]})
        return results
