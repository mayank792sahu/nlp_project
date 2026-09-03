"""Pairwise statistical significance testing between models using McNemar's exact test
on paired correct/incorrect outcomes over the shared test set, so accuracy differences
in the comparison table can be backed by more than "the number is bigger"."""

from itertools import combinations
from typing import Sequence

import pandas as pd
from scipy import stats

from finsent.config import CONFIG, resolve_path


def _tables_dir():
    d = resolve_path(CONFIG["results"]["tables_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def mcnemar_exact(y_true: Sequence[str], preds_a: Sequence[str], preds_b: Sequence[str]) -> dict:
    """Exact two-sided McNemar's test, binarized on correct/incorrect per sentence.

    Only the discordant pairs (one model right, the other wrong) carry information;
    under the null hypothesis they split 50/50 between the two models, tested via an
    exact binomial test (robust regardless of sample size, unlike the chi-square
    approximation).
    """
    correct_a = [t == p for t, p in zip(y_true, preds_a)]
    correct_b = [t == p for t, p in zip(y_true, preds_b)]

    n_both_correct = sum(ca and cb for ca, cb in zip(correct_a, correct_b))
    n_both_wrong = sum((not ca) and (not cb) for ca, cb in zip(correct_a, correct_b))
    n_a_only = sum(ca and not cb for ca, cb in zip(correct_a, correct_b))  # a correct, b wrong
    n_b_only = sum(cb and not ca for ca, cb in zip(correct_a, correct_b))  # b correct, a wrong

    discordant = n_a_only + n_b_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = stats.binomtest(min(n_a_only, n_b_only), discordant, 0.5, alternative="two-sided").pvalue

    return {
        "n_both_correct": n_both_correct,
        "n_a_only_correct": n_a_only,
        "n_b_only_correct": n_b_only,
        "n_both_wrong": n_both_wrong,
        "p_value": p_value,
    }


def pairwise_significance(y_true: Sequence[str], predictions: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Runs McNemar's test on every model pair; saves + returns the comparison table."""
    rows = []
    for model_a, model_b in combinations(predictions.keys(), 2):
        result = mcnemar_exact(y_true, predictions[model_a], predictions[model_b])
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            **result,
            "significant": result["p_value"] < alpha,
        })

    df = pd.DataFrame(rows)
    df.to_csv(_tables_dir() / "significance.csv", index=False)
    return df
