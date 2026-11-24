# warmup_02.py
# Week 2 warmup exercises: scikit-learn and linear regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- scikit-learn API ---

# Q1
print("--- scikit-learn Q1 ---")

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

# step 1: create the model
model = LinearRegression()

# step 2: fit the model to the data
model.fit(temp_min, temp_max)

# step 3: predict for a low of 6 and a low of 18
# (predict needs a 2D input, so I use double square brackets)
pred_6 = model.predict([[6]])
pred_18 = model.predict([[18]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6:", pred_6[0])
print("Predicted high for a low of 18:", pred_18[0])

# Q2
print("\n--- scikit-learn Q2 ---")

x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("New shape after reshape(-1, 1):", x_2d.shape)

# scikit-learn expects X to be 2D because it thinks of X as a table:
# each row is one sample (one day) and each column is one feature.
# A 1D array has no columns, so scikit-learn can't tell if it is
# 5 samples with 1 feature or 1 sample with 5 features.
# reshape(-1, 1) makes it 5 rows and 1 column.


# --- Linear Regression ---

# Setup data (used for all the Linear Regression questions)
rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
print("\n--- Linear Regression Q1 ---")

plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")
plt.savefig("outputs/high_vs_low.png")
plt.close()
print("Saved outputs/high_vs_low.png")

# What I see: the points go up in a straight line (warmer lows mean warmer highs).
# There are two visible groups, one a bit above the other. The rainy days (red)
# sit lower than the dry days (blue). This suggests is_rainy matters:
# on rainy days the high is lower for the same low temperature.

# Q2
print("\n--- Linear Regression Q2 ---")

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
print("\n--- Linear Regression Q3 ---")

model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R² on the test set:", model.score(X_test, y_test))

# The slope is about 0.94. That means for every 1 degree the daily low goes up,
# the daily high goes up by about 0.94 degrees too (almost 1 for 1).

# Q4
print("\n--- Linear Regression Q4 ---")

X_full = np.column_stack([temp_min, is_rainy])

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

r2_one_feature = model.score(X_test, y_test)
r2_two_features = model_full.score(X_test_f, y_test_f)

print("Test R² (temp_min only):", r2_one_feature)
print("Test R² (temp_min + is_rainy):", r2_two_features)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Yes, adding the rain flag helps. The test R² goes up from about 0.894 to
# about 0.945, so the model explains more of the variation in the daily high.
# The is_rainy coefficient is about -2.9. In practical terms, if two days have the
# same low temperature, the rainy day is predicted to have a high about 3 degrees
# lower than the dry day.

# Q5
print("\n--- Linear Regression Q5 ---")

y_pred_full = model_full.predict(X_test_f)

plt.scatter(y_pred_full, y_test_f)

# diagonal line where predicted = actual
low = min(y_pred_full.min(), y_test_f.min())
high = max(y_pred_full.max(), y_test_f.max())
plt.plot([low, high], [low, high], color="red")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High")
plt.ylabel("Actual High")
plt.savefig("outputs/predicted_vs_actual_warmup.png")
plt.close()
print("Saved outputs/predicted_vs_actual_warmup.png")

# A point above the diagonal means the actual high was bigger than the model
# predicted (the model under-predicted). A point below the diagonal means the
# actual high was smaller than predicted (the model over-predicted).