"""Qualitative error analysis: per-model misclassifications, neutral-class confusions
specifically, and sentences that are hard across multiple models — the evidence base
for the "why do lexicon methods fail on financial hedging language" discussion."""

from typing import Sequence

import pandas as pd

from finsent.config import CONFIG, resolve_path


def _tables_dir():
    d = resolve_path(CONFIG["results"]["tables_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(model_name: str) -> str:
    return model_name.replace(" ", "_").replace("(", "").replace(")", "")


def misclassified_examples(sentences: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame({"sentence": sentences, "true_label": y_true, "predicted_label": y_pred})
    return df[df["true_label"] != df["predicted_label"]].reset_index(drop=True)


def neutral_class_errors(sentences: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Misclassifications where neutral is involved in either direction (true or predicted)."""
    df = misclassified_examples(sentences, y_true, y_pred)
    return df[(df["true_label"] == "neutral") | (df["predicted_label"] == "neutral")].reset_index(drop=True)


def hardest_sentences(
    sentences: Sequence[str],
    y_true: Sequence[str],
    predictions: dict,
    min_models_wrong: int = 2,
) -> pd.DataFrame:
    """Sentences misclassified by at least `min_models_wrong` of the given models,
    with every model's prediction attached, sorted by how many models got it wrong."""
    rows = []
    for i, (sentence, true_label) in enumerate(zip(sentences, y_true)):
        wrong_models = [name for name, preds in predictions.items() if preds[i] != true_label]
        if len(wrong_models) >= min_models_wrong:
            row = {"sentence": sentence, "true_label": true_label, "n_models_wrong": len(wrong_models)}
            for name, preds in predictions.items():
                row[f"pred_{name}"] = preds[i]
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("n_models_wrong", ascending=False).reset_index(drop=True)
    return df


def generate_error_analysis(sentences: Sequence[str], y_true: Sequence[str], predictions: dict) -> dict:
    """Writes per-model misclassification + neutral-error CSVs and a cross-model
    hardest-sentences CSV to results/tables/, and returns them as DataFrames."""
    tables_dir = _tables_dir()
    out = {}

    for name, preds in predictions.items():
        safe = _safe_name(name)

        errors_df = misclassified_examples(sentences, y_true, preds)
        errors_df.to_csv(tables_dir / f"errors_{safe}.csv", index=False)
        out[f"errors_{name}"] = errors_df

        neutral_df = neutral_class_errors(sentences, y_true, preds)
        neutral_df.to_csv(tables_dir / f"neutral_errors_{safe}.csv", index=False)
        out[f"neutral_errors_{name}"] = neutral_df

    hardest_df = hardest_sentences(sentences, y_true, predictions)
    hardest_df.to_csv(tables_dir / "hardest_sentences.csv", index=False)
    out["hardest_sentences"] = hardest_df

    return out
