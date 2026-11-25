from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@dataclass
class Prediction:
    """Result for one day. probability is the chance the day is good."""

    label: str
    probability: float


class WeatherClassifier:
    """Predicts if a day is good for running."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the saved model.

        Raises:
            FileNotFoundError: if model_path is not a file.
        """
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"model file not found: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, days: list[dict]) -> list[Prediction]:
        """Return one Prediction per day, in the same order.

        Raises:
            ValueError: if a day is missing a feature.
        """
        if not days:
            return []

        for day in days:
            missing = [f for f in FEATURES if f not in day]
            if missing:
                raise ValueError(f"missing features: {missing}")

        X = pd.DataFrame(days)[FEATURES]
        labels = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        return [
            Prediction("good" if label == 1 else "skip", float(prob))
            for label, prob in zip(labels, probs)
        ]
