import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from weather_model import Prediction, WeatherClassifier
from weather_model.classifier import FEATURES


# break check: always returning "skip" made test_labels (good day)
# and test_three_days_in_order fail


def make_day(high, low, rain, wind):
    return dict(zip(FEATURES, [high, low, rain, wind]))


GOOD = make_day(20, 12, 0, 8)
BAD = make_day(3, -2, 15, 40)


@pytest.fixture
def clf(tmp_path):
    # tiny model, 1 = good
    X = pd.DataFrame([GOOD, make_day(22, 14, 0, 10), BAD, make_day(1, -4, 20, 35)])
    model = LogisticRegression().fit(X, [1, 1, 0, 0])
    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    return WeatherClassifier(path)


@pytest.mark.parametrize("day, expected", [(GOOD, "good"), (BAD, "skip")])
def test_labels(clf, day, expected):
    assert clf.predict([day])[0].label == expected


def test_probability_range(clf):
    assert 0 <= clf.predict([GOOD])[0].probability <= 1


def test_three_days_in_order(clf):
    results = clf.predict([GOOD, BAD, GOOD])
    assert all(isinstance(r, Prediction) for r in results)
    assert [r.label for r in results] == ["good", "skip", "good"]


def test_empty_list(clf):
    assert clf.predict([]) == []


def test_missing_feature(clf):
    day = GOOD.copy()
    del day["precipitation_sum"]
    with pytest.raises(ValueError, match="precipitation_sum"):
        clf.predict([day])


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        WeatherClassifier(tmp_path / "nope.pkl")
