from finsent.evaluation.error_analysis import (
    hardest_sentences,
    misclassified_examples,
    neutral_class_errors,
)


def test_misclassified_examples_filters_correctly():
    sentences = ["a", "b", "c"]
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "positive", "neutral"]
    df = misclassified_examples(sentences, y_true, y_pred)
    assert len(df) == 1
    assert df.iloc[0]["sentence"] == "b"


def test_neutral_class_errors_includes_both_directions():
    sentences = ["a", "b", "c"]
    y_true = ["neutral", "positive", "negative"]
    y_pred = ["positive", "neutral", "positive"]
    df = neutral_class_errors(sentences, y_true, y_pred)
    # "a": neutral -> positive, "b": positive -> neutral both involve neutral; "c" does not
    assert len(df) == 2
    assert set(df["sentence"]) == {"a", "b"}


def test_hardest_sentences_flags_multi_model_failures():
    sentences = ["x", "y"]
    y_true = ["positive", "negative"]
    predictions = {
        "m1": ["negative", "negative"],  # wrong on x, correct on y
        "m2": ["negative", "positive"],  # wrong on x, wrong on y
    }
    df = hardest_sentences(sentences, y_true, predictions, min_models_wrong=2)
    assert len(df) == 1
    assert df.iloc[0]["sentence"] == "x"
    assert df.iloc[0]["n_models_wrong"] == 2
