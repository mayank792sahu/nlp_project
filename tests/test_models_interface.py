from finsent.models.base import LABELS
from finsent.models.vader_model import VaderModel


def test_vader_predict_returns_canonical_labels():
    model = VaderModel()
    preds = model.predict(["Profits soared this quarter.", "The stock crashed badly.", "Meeting at 3pm."])
    assert all(p in LABELS for p in preds)


def test_vader_fit_is_noop_and_returns_self():
    model = VaderModel()
    result = model.fit(["some text"], ["positive"])
    assert result is model
