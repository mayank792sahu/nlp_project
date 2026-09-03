"""Canonical label set shared across data loading, models, and evaluation."""

LABELS = ("negative", "neutral", "positive")

# Financial PhraseBank's own int encoding (HF `datasets` ClassLabel order).
PHRASEBANK_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
