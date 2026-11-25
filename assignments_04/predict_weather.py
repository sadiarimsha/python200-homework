from pathlib import Path

from weather_model import WeatherClassifier

MODEL_PATH = Path(__file__).parent / "models" / "weather_classifier.pkl"


def main():
    clf = WeatherClassifier(MODEL_PATH)

    keys = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
    values = [
        [20, 12, 0, 8],  # good
        [3, -2, 15, 40],  # bad
        [28, 19, 1, 18],  # borderline warm
        [9, 3, 2, 22],  # borderline cool
    ]
    days = [dict(zip(keys, v)) for v in values]

    for day, result in zip(days, clf.predict(days)):
        print(
            f"high {day[keys[0]]}C, low {day[keys[1]]}C, "
            f"rain {day[keys[2]]}mm, wind {day[keys[3]]}km/h -> "
            f"{result.label} ({result.probability:.2f})"
        )


# without this guard, importing this file would run main() too
if __name__ == "__main__":
    main()

# Reflection
# 1. if a caller sends wind and rain in a different order, the model could read
#    wind as rain and give wrong answers. a fixed order prevents this.
# 2. the tests run without my real file, and retraining it won't change them.
# 3. ready: it loads the model once and predicts a batch.
#    to add: check for None or non-number values from the database.
