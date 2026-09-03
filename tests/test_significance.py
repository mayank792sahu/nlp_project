import finsent.evaluation.significance as significance
from finsent.evaluation.significance import mcnemar_exact, pairwise_significance


def test_mcnemar_identical_predictors_not_significant():
    y_true = ["positive"] * 10 + ["negative"] * 10
    preds = list(y_true)  # both models identical and perfect -> no discordant pairs
    result = mcnemar_exact(y_true, preds, preds)
    assert result["p_value"] == 1.0
    assert result["n_a_only_correct"] == 0
    assert result["n_b_only_correct"] == 0


def test_mcnemar_detects_clear_difference():
    y_true = ["positive"] * 30
    preds_a = ["positive"] * 30  # always correct
    preds_b = ["negative"] * 30  # always wrong
    result = mcnemar_exact(y_true, preds_a, preds_b)
    assert result["p_value"] < 0.05
    assert result["n_a_only_correct"] == 30
    assert result["n_b_only_correct"] == 0


def test_pairwise_significance_covers_all_pairs(tmp_path, monkeypatch):
    # pairwise_significance writes a CSV as a side effect; redirect it to a scratch dir
    # so this test doesn't clobber the project's real results/tables/significance.csv.
    monkeypatch.setattr(significance, "_tables_dir", lambda: tmp_path)

    y_true = ["positive", "negative", "neutral"] * 5
    predictions = {
        "A": list(y_true),
        "B": ["neutral"] * len(y_true),
    }
    df = pairwise_significance(y_true, predictions)
    assert len(df) == 1
    assert set(df.iloc[0][["model_a", "model_b"]]) == {"A", "B"}
    assert (tmp_path / "significance.csv").exists()
