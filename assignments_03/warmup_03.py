import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---

# Q1
print("--- Preprocessing Q1 ---")
scaler = StandardScaler()
scaler.fit(X_train)                      # learn mean and std from the training data only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# mean of each column of the scaled training data (should be about 0)
for i in range(len(numeric)):
    print(numeric[i], "mean:", X_train_scaled[:, i].mean())

# I fit the scaler on X_train only so no information from the test set leaks into training.

# Q2
print("--- Preprocessing Q2 ---")
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])
print("Shape:", season_encoded.shape)
print("Category names:", encoder.get_feature_names_out())

# The single season column became 4 columns (fall, spring, summer, winter). The
# number is 4 because one-hot encoding makes one new 0/1 column for each unique
# category, and there are four unique seasons.

# Q3
print("--- Preprocessing Q3 ---")
features_with_season = numeric + ["season"]
X2 = df[features_with_season]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(), ["season"]),
])

pipe_season = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe_season.fit(X2_train, y2_train)
print("Test accuracy:", pipe_season.score(X2_test, y2_test))

# The pipeline automatically scales the numeric columns and one-hot encodes season
# (fitting on the training data only), and then runs the same steps on new data
# before predicting. By hand I would have to do each of those steps myself for
# the train set and again for the test set, and keep them in the right order.

# --- KNN ---

# Q1
print("--- KNN Q1 ---")
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_pipe.fit(X_train, y_train)
knn_pred = knn_pipe.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# Q2
print("--- KNN Q2 ---")
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
best_k = None
best_score = 0
for k in k_values:
    pipe_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(pipe_k, X_train, y_train, cv=5, scoring="accuracy")
    mean_score = scores.mean()
    print("k =", k, " mean CV accuracy =", round(mean_score, 4))
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
print("Best k:", best_k)

# I would choose k=11. It had the highest mean CV accuracy (0.9417). k=7 (0.9396)
# and k=15 (0.9375) were very close, so the exact choice is not a big deal, but
# a mid-sized k is less likely to be thrown off by a few noisy days. k=1 was
# lower (0.925) because it overfits to single days.

# Q3
print("--- KNN Q3 ---")
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)                    # raw, unscaled data
print("Accuracy without scaling:", knn_raw.score(X_test, y_test))

knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
print("Accuracy with scaling:   ", knn_scaled.score(X_test, y_test))

# Scaling helped: 0.9583 with scaling vs 0.925 without. Scaling matters for KNN because it works by measuring distance. A column with
# big numbers (like temperature) can swamp a column with small numbers (like
# precipitation), so the small-number columns barely count unless we put every
# column on the same scale.

# --- Logistic Regression ---

# Q1
print("--- Logistic Regression Q1 ---")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)
lr_pred = pipeline.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:   ", recall_score(y_test, lr_pred))
print("F1:       ", f1_score(y_test, lr_pred))

# Q2
print("--- Logistic Regression Q2 ---")
coefs = pipeline.named_steps["logreg"].coef_[0]
for i in range(len(numeric)):
    print(numeric[i], ":", round(coefs[i], 3))

# Negative coefficients push toward "skip" and positive ones push toward "good".
# Wind (-2.444) and precipitation (-1.889) push most strongly toward "skip", which
# matches my intuition: windy and rainy days are bad for running. The temperature
# coefficients are small (max -0.121, min +0.256). That makes sense because a day
# can be too cold OR too hot, and a single straight-line model cannot capture
# "not too hot, not too cold", so temperature ends up with little effect.

# Q3
print("--- Logistic Regression Q3 ---")
probs = pipeline.predict_proba(X_test)[:, 1]     # column 1 = probability of "good"
for i in range(5):
    print("Test day", i + 1, "probability of good:", round(probs[i], 2))

# A probability near 0.5 means the model is basically unsure, because the day
# sits right on the boundary between "good" and "skip". For example, test day 2
# got 0.47, so the model is nearly guessing. Test day 4 got 0.96 and test day 1
# got 0.04, so the model is confident about those.

# --- Evaluation ---

# Q1
print("--- Evaluation Q1 ---")
cm = confusion_matrix(y_test, lr_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

# In the matrix the layout is [[true skip & predicted skip, false positives],
#                              [false negatives, true good & predicted good]].
false_positives = cm[0][1]
false_negatives = cm[1][0]
print("False positives:", false_positives)
print("False negatives:", false_negatives)

# The model made 9 false positives and 7 false negatives. For a running app I
# think false positives matter more: sending someone out on a bad day (storm, freezing cold) is worse than
# missing a good day, which only costs them a run they could have taken anyway.

# --- joblib ---

# Q1
print("--- joblib Q1 ---")
joblib.dump(pipeline, "models/warmup_model.pkl")
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")

# If I had saved only the LogisticRegression step without the scaler, then called
# .predict() on raw data, the model would get unscaled numbers that it was never
# trained on. The coefficients expect scaled values, so the predictions would be
# wrong and there would be no error to warn me. Saving the whole pipeline keeps
# the scaler and the model together.
