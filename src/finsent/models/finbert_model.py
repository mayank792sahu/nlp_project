"""FinBERT (ProsusAI/finbert): pretrained transformer, zero-shot inference only
(no fine-tuning), evaluated as a reference point. Runs batched on CPU."""

from typing import Sequence

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from finsent.config import CONFIG
from finsent.models.base import SentimentModel


class FinBertModel(SentimentModel):
    name = "FinBERT (zero-shot)"

    def __init__(self):
        cfg = CONFIG["finbert"]
        checkpoint = cfg["checkpoint"]
        self._batch_size = cfg["batch_size"]

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        # Map the model's own id2label (order may differ from the canonical LABELS) to lowercase strings.
        self._id2label = {i: label.lower() for i, label in model.config.id2label.items()}

        self._pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=cfg["device"],
        )

    def predict(self, texts: Sequence[str]) -> list:
        outputs = self._pipeline(list(texts), batch_size=self._batch_size, truncation=True)
        return [out["label"].lower() for out in outputs]

    def predict_proba(self, texts: Sequence[str]) -> list:
        outputs = self._pipeline(list(texts), batch_size=self._batch_size, truncation=True, top_k=None)
        return [{item["label"].lower(): item["score"] for item in out} for out in outputs]
