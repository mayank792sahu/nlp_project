import pytest

from finsent.config import CONFIG, resolve_path

LM_CSV_PATH = resolve_path(CONFIG["lexicons"]["loughran_mcdonald_csv"])

pytestmark = pytest.mark.skipif(
    not LM_CSV_PATH.exists(),
    reason=f"Loughran-McDonald dictionary not found at {LM_CSV_PATH}; download it manually (see README).",
)


def test_lm_scorer_detects_positive_sentence():
    from finsent.models.loughran_mcdonald import LoughranMcDonaldModel

    model = LoughranMcDonaldModel()
    preds = model.predict(["The company reported strong profit and excellent growth."])
    assert preds[0] == "positive"


def test_lm_scorer_detects_negative_sentence():
    from finsent.models.loughran_mcdonald import LoughranMcDonaldModel

    model = LoughranMcDonaldModel()
    preds = model.predict(["The company reported a significant loss and severe decline."])
    assert preds[0] == "negative"


def test_lm_scorer_neutral_when_no_sentiment_words():
    from finsent.models.loughran_mcdonald import LoughranMcDonaldModel

    model = LoughranMcDonaldModel()
    preds = model.predict(["The meeting is scheduled for Tuesday afternoon."])
    assert preds[0] == "neutral"
