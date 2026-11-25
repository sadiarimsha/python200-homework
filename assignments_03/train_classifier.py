import json
import sys
import requests
import pandas as pd
import sklearn
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

# ---------------------------------------------------------------
# Task 1: Fetch the Data
# ---------------------------------------------------------------
# City I chose: Chicago, Illinois
city = "Chicago, IL"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 41.88,
    "longitude": -87.63,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Chicago",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# Task 2: Engineer the Label
# ---------------------------------------------------------------
# I kept the thresholds exactly as given in the assignment (no changes):
# high between 7 and 26, low at least 0, precipitation under 3 mm, wind under 30 km/h.
df = df.dropna()   # drop any days with missing values so the models do not crash

good = (
    (df["temperature_2m_max"] >= 7)
    & (df["temperature_2m_max"] <= 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
)
df["good_for_running"] = good.astype(int)

print("\nClass balance (1 = good, 0 = skip):")
print(df["good_for_running"].value_counts())
print("Fraction good:", round(df["good_for_running"].mean(), 3))

# About 34% of days are good (125 good, 240 skip). That seems reasonable for
# Chicago: cold winters (lows below 0, highs under 7), some hot summer days, and
# rain or wind on many other days. The data is somewhat imbalanced, so accuracy
# alone can look okay just by guessing "skip" a lot. Precision, recall and F1 for
# the "good" class matter more.

# ---------------------------------------------------------------
# Task 3: Build and Evaluate a KNN Classifier
# ---------------------------------------------------------------
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- KNN: choosing k with 5-fold cross-validation ---")
# I start at k=5 on purpose: with k=1 or 3 the predicted probabilities can only be
# a few values like 0 or 1, which makes the confidence score in predict.py useless.
k_values = [5, 7, 9, 11, 15, 21]
best_k = None
best_score = 0
for k in k_values:
    knn_try = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn_try, X_train, y_train, cv=5, scoring="accuracy")
    print("k =", k, " mean CV accuracy =", round(scores.mean(), 4))
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k
print("Chosen k:", best_k)

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

print("\nKNN test accuracy:", round(knn_acc, 4))
print(classification_report(y_test, knn_pred))

# ---------------------------------------------------------------
# Task 4: Build and Evaluate a Logistic Regression Classifier
# ---------------------------------------------------------------
logreg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
logreg_model.fit(X_train, y_train)
lr_pred = logreg_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("--- Logistic Regression ---")
print("Accuracy: ", round(lr_acc, 4))
print("Precision:", round(precision_score(y_test, lr_pred), 4))
print("Recall:   ", round(recall_score(y_test, lr_pred), 4))
print("F1:       ", round(lr_f1, 4))
print(classification_report(y_test, lr_pred))

print("Coefficients:")
coefs = logreg_model.named_steps["logreg"].coef_[0]
for i in range(len(features)):
    print(" ", features[i], ":", round(coefs[i], 3))

# Negative coefficients push a day toward "skip". Precipitation (-2.897) pushes
# hardest toward "skip", then wind (-0.511). Rain matters most in Chicago because
# the rule needs under 3 mm and a rainy day is almost always a skip. The
# temperature coefficients are less clear (max -0.112, min +0.742) because a day
# can be too cold OR too hot, and one straight boundary cannot capture both sides
# of that.

# ---------------------------------------------------------------
# Task 5: Compare and Choose
# ---------------------------------------------------------------
print("\n--- Comparison ---")
print("KNN accuracy:               ", round(knn_acc, 4), " F1:", round(knn_f1, 4))
print("Logistic Regression accuracy:", round(lr_acc, 4), " F1:", round(lr_f1, 4))

# 1. KNN had higher accuracy: 0.8219 vs 0.7397 for logistic regression. KNN also
#    had a higher F1 for the "good" class (0.7719 vs 0.5778).
#
# 2. For a running app, accuracy is not the only thing I care about. A false
#    positive means the app says "good" on a bad day and sends someone out in
#    freezing, wet or windy weather. A false negative means it says "skip" on a
#    good day and the person misses a run. False positives are worse, so I weigh
#    precision for the "good" class most, while keeping recall reasonable so the
#    app is not too cautious to be useful.
#
# 3. I am deploying KNN (k=9). The code picks whichever model has the higher F1
#    score on the test set, and KNN won clearly. Details of my reasoning:
#    (F1 balances precision and recall, which matters because the classes are
#    imbalanced). The "good day" rule is a box: the high has to be inside a range,
#    AND the low, rain and wind all have to be inside their limits. Logistic
#    regression draws one straight boundary, so it has a hard time with a box and
#    tends to miss good days (low recall). KNN looks at similar days, so it can
#    follow the box shape better. The tradeoffs: logistic regression is smaller to
#    save, faster, easier to explain (coefficients) and gives smooth probabilities;
#    KNN has to store all the training days and compare each new day to them.
#    KNN won by a clear margin (F1 0.77 vs 0.58), so accuracy and F1 matter more
#    to me than the smaller file size. The downsides I accept: a larger saved
#    model, slower predictions, and no coefficients to explain a decision. Its
#    probabilities are also less smooth (with k=9 they come in steps of 1/9).
#    One caution: KNN's precision for "good" is only 0.69, so about 3 in 10 "good"
#    predictions are wrong. That is the error I care most about for a running app.

if knn_f1 >= lr_f1:
    chosen_model = knn_model
    chosen_acc = knn_acc
    chosen_f1 = knn_f1
    chosen_name = "StandardScaler + KNeighborsClassifier (k=" + str(best_k) + ")"
else:
    chosen_model = logreg_model
    chosen_acc = lr_acc
    chosen_f1 = lr_f1
    chosen_name = "StandardScaler + LogisticRegression"

print("Deploying:", chosen_name)

# ---------------------------------------------------------------
# Task 6: Save the Model
# ---------------------------------------------------------------
joblib.dump(chosen_model, "models/weather_classifier.pkl")

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "feature_names": features,
    "label_rule": "good_for_running = 1 if 7 <= temperature_2m_max <= 26 and temperature_2m_min >= 0 and precipitation_sum < 3 and wind_speed_10m_max < 30, else 0",
    "city": city,
    "model_type": chosen_name,
    "test_accuracy": round(float(chosen_acc), 4),
    "test_f1": round(float(chosen_f1), 4),
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved models/weather_classifier.pkl and models/weather_classifier_metadata.json")