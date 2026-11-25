import json
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Task 7: Load and Predict
# ---------------------------------------------------------------
# This script only loads the saved model. No training, no API calls.

model = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

print("City:", metadata["city"])
print("Model:", metadata["model_type"])
print("Features:", metadata["feature_names"])
print("Test accuracy:", metadata["test_accuracy"])
print()

# Hypothetical days. Column names come from the metadata so they match training.
feature_names = metadata["feature_names"]
days = pd.DataFrame(
    [
        [18, 9, 0.0, 10],    # clearly good: mild, dry, calm
        [-8, -15, 0.0, 15],  # clearly bad: freezing
        [8, 1, 0.0, 12],     # borderline: right at the cold edge of "good"
        [25, 14, 2.5, 28],   # borderline: warm, almost 3 mm rain, almost 30 km/h wind
        [15, 8, 12.0, 40],   # clearly bad: rainy and windy
    ],
    columns=feature_names,
)

predictions = model.predict(days)
probabilities = model.predict_proba(days)[:, 1]   # probability of "good"

for i in range(len(days)):
    if predictions[i] == 1:
        label = "good"
    else:
        label = "skip"
    print("Day", i + 1)
    print("  Inputs:", days.iloc[i].to_dict())
    print("  Prediction:", label)
    print("  Confidence (probability of good):", round(probabilities[i], 2))
    print()

# ---------------------------------------------------------------
# Task 8: Reflect
# ---------------------------------------------------------------
# 1. My borderline day is Day 3 (high 8, low 1, no rain, light wind), which is
#    only just inside the "good" rule. The model gave it a 0.33 probability of
#    good and predicted "skip". I would call the model uncertain here, not
#    confident, and by my own rule this day is actually good, so the model got it
#    wrong. A probability near 0.5 means the model is uncertain, while one near 0
#    or 1 means it is confident. (With k=9, probabilities come in steps of 1/9,
#    so 0.52 itself cannot happen, but 0.44 or 0.56 would be the same situation.)
#    For a day where it predicts about 0.52, I would not trust the "good" label. I
#    would tell the user "borderline, check the forecast yourself" instead of a
#    firm yes or no, or require a higher cutoff (like 0.7) before saying "good",
#    since sending someone out on a bad day (false positive) is the worse error.
#
# 2. If predict.py runs before train_classifier.py, joblib.load() would crash
#    with a FileNotFoundError because models/weather_classifier.pkl does not
#    exist yet. A more helpful message would catch that error and print
#    something like: "Model file not found. Run train_classifier.py first to
#    create models/weather_classifier.pkl."
#
# 3. In Week 10 the pipeline classifies each day automatically, but my
#    predict.py uses hard-coded hypothetical days. It would need to take real
#    weather data as input (for example from the API or a file) instead of
#    typing the numbers in by hand, and it would need to save or send the
#    results somewhere instead of just printing them.