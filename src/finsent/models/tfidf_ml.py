"""Supervised ML approach: TF-IDF features + Logistic Regression or Linear SVM,
trained directly on labeled financial sentences (preprocessed, negation-safe form)."""

from typing import Sequence

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from finsent.config import CONFIG, resolve_path
from finsent.models.base import SentimentModel
from finsent.preprocessing.text_cleaning import clean_for_lexicon_ml_str

_CLASSIFIERS = {
    "logreg": lambda C, seed: LogisticRegression(C=C, max_iter=1000, random_state=seed),
    "linear_svc": lambda C, seed: LinearSVC(C=C, random_state=seed),
}


class TfidfMlModel(SentimentModel):
    name = "TFIDF+ML"

    def __init__(self):
        self._pipeline: Pipeline = None

    def _build_pipeline(self, classifier_key: str, C: float) -> Pipeline:
        cfg = CONFIG["tfidf_ml"]
        seed = CONFIG["seed"]
        vectorizer = TfidfVectorizer(
            preprocessor=clean_for_lexicon_ml_str,
            ngram_range=tuple(cfg["ngram_range"]),
            min_df=cfg["min_df"],
        )
        clf = _CLASSIFIERS[classifier_key](C, seed)
        return Pipeline([("tfidf", vectorizer), ("clf", clf)])

    def fit(self, texts: Sequence[str], labels: Sequence[str],
            val_texts: Sequence[str] = None, val_labels: Sequence[str] = None) -> "TfidfMlModel":
        """Light hyperparameter sweep over classifier type and C, selected on val macro-F1
        if a validation set is provided; otherwise defaults to the first classifier/C."""
        cfg = CONFIG["tfidf_ml"]
        candidates = [
            (clf_key, C) for clf_key in cfg["classifiers"] for C in cfg["C_values"]
        ]

        best_pipeline, best_score, best_name = None, -1.0, None
        for clf_key, C in candidates:
            pipeline = self._build_pipeline(clf_key, C)
            pipeline.fit(texts, labels)

            if val_texts is not None and val_labels is not None:
                preds = pipeline.predict(val_texts)
                score = f1_score(val_labels, preds, average="macro", zero_division=0)
            else:
                score = 0.0

            if score >= best_score:
                best_pipeline, best_score, best_name = pipeline, score, f"{clf_key}(C={C})"

        self._pipeline = best_pipeline
        self.selected_config = best_name
        return self

    def predict(self, texts: Sequence[str]) -> list:
        if self._pipeline is None:
            raise RuntimeError("TfidfMlModel.fit() must be called before predict().")
        return list(self._pipeline.predict(texts))

    def predict_proba(self, texts: Sequence[str]):
        """Returns per-class probabilities if the underlying classifier supports it
        (LogisticRegression does; LinearSVC does not and this returns None)."""
        if self._pipeline is None:
            raise RuntimeError("TfidfMlModel.fit() must be called before predict_proba().")
        clf = self._pipeline.named_steps["clf"]
        if not hasattr(clf, "predict_proba"):
            return None
        proba = self._pipeline.predict_proba(texts)
        return [dict(zip(clf.classes_, row)) for row in proba]

    def save(self, path: str = None) -> None:
        out_path = resolve_path(path or CONFIG["tfidf_ml"]["model_out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, out_path)

    def load(self, path: str = None) -> "TfidfMlModel":
        in_path = resolve_path(path or CONFIG["tfidf_ml"]["model_out"])
        self._pipeline = joblib.load(in_path)
        return self
