"""Common interface all sentiment models implement, so the evaluation harness
can treat lexicon rules, a trained sklearn pipeline, and a pretrained
transformer uniformly."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from finsent.labels import LABELS

__all__ = ["LABELS", "SentimentModel"]


class SentimentModel(ABC):
    name: str = "SentimentModel"

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SentimentModel":
        """No-op for lexicon/zero-shot models; overridden by models that actually train."""
        return self

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> list:
        """Return one label from LABELS per input text."""
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> Optional[list]:
        """Optional: list of {label: score} dicts. None if unsupported."""
        return None
