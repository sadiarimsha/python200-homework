# project_02.py
# Week 2 mini-project: predicting the daily high temperature

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Task 1: Fetch the Data ---
print("--- Task 1: Fetch the Data ---")

# City I chose: Chicago, IL
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

print("Shape of the dataset:", df.shape)
print(df.head())


# --- Task 2: Look at the Features ---
print("\n--- Task 2: Look at the Features ---")

print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())
# There are no missing values (all columns show 0), so I did not need to
# drop or fill anything.

# correlation of each numeric column with the daily high
numeric_df = df.select_dtypes(include="number")
correlations = numeric_df.corr()["temperature_2m_max"].sort_values()
print("\nCorrelation with temperature_2m_max:")
print(correlations)
# temperature_2m_min has the strongest relationship with the daily high
# (0.93). Wind speed is weakly negative (-0.21) and precipitation is weakly
# positive (0.11).

# plot: scatter of low vs high
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])
plt.title("Daily Low vs Daily High")
plt.xlabel("Daily Low Temperature (C)")
plt.ylabel("Daily High Temperature (C)")
plt.savefig("outputs/low_vs_high.png")
plt.close()
print("Saved outputs/low_vs_high.png")

# The plot shows a strong straight-line pattern. The few very cold days at the
# bottom left (lows around -17 to -20 C) are real winter days, not errors.

# --- Task 3: Engineer a Feature ---
print("\n--- Task 3: Engineer a Feature ---")

month = df["date"].dt.month
df["is_summer"] = ((month == 6) | (month == 7) | (month == 8)).astype(int)

print("Summer days:", (df["is_summer"] == 1).sum())
print("Non-summer days:", (df["is_summer"] == 0).sum())
# I expected is_summer to help only a little, because the daily low already
# captures most of the seasonal warmth. The results below show it added almost nothing.


# --- Task 4: Baseline Model ---
print("\n--- Task 4: Baseline Model ---")

X_base = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred_base = baseline_model.predict(X_test)
base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
base_r2 = r2_score(y_test, y_pred_base)

print("Slope:", baseline_model.coef_[0])
print("RMSE:", base_rmse)
print("R² on the test set:", base_r2)
# The RMSE is about 3.75 C, so a typical prediction is off by about 3.75 degrees
# Celsius (roughly 6.8 F). R² is 0.877, so the low temperature alone explains
# about 88% of the variation in the high.

# --- Task 5: Full Model ---
print("\n--- Task 5: Full Model ---")

feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
y_pred = model.predict(X_test)
full_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("RMSE on the test set:", full_rmse)
print("Baseline test R² was:", base_r2)
print("Improvement in test R²:", test_r2 - base_r2)
# The full model's test R² (0.8763) is basically the same as the baseline
# (0.8768), so adding the three extra features did not help.

print("\nCoefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")
# temperature_2m_min (+0.993) is by far the main driver: each extra degree in
# the low adds about 1 degree to the high. is_summer (+0.784) raises the
# predicted high a bit. precipitation_sum (-0.069) lowers it slightly, and
# wind_speed_10m_max (+0.006) is close to zero. The surprising signs are
# precipitation (correlation was positive, coefficient is negative) and wind
# (correlation was negative, coefficient is about zero).

# Train R² is 0.861 and test R² is 0.876. They are close, and the test score is
# even a little higher, so there is no overfitting. The difference is likely just
# luck in which 73 days ended up in the test set.


# --- Task 6: Evaluate and Summarize ---
print("\n--- Task 6: Evaluate and Summarize ---")

plt.scatter(y_pred, y_test)
low = min(y_pred.min(), y_test.min())
high = max(y_pred.max(), y_test.max())
plt.plot([low, high], [low, high], color="red")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted High (C)")
plt.ylabel("Actual High (C)")
plt.savefig("outputs/predicted_vs_actual_high.png")
plt.close()
print("Saved outputs/predicted_vs_actual_high.png")
# The error is not the same across the whole range. The model is most accurate
# on very cold days (low end), where the points sit close to the red line.
# The biggest errors are in the middle range (predicted highs of about 5 to 15),
# where some days were much warmer than predicted. At the high end the model
# slightly over-predicts, with most points just below the line.

print("\nDataset size:", len(df), "rows")
print("Test set size:", len(y_test), "rows")

# Summary:
# - The dataset has 365 days of Chicago weather in 2023. The test set has 73 days.
# - The full model has RMSE = 3.76 C and R² = 0.876. A typical error of about
#   3.8 C (about 6.8 F) means a predicted high of 20 C is usually within about
#   16 to 24 C of the real value.
# - The feature with the largest effect is temperature_2m_min, and it pushes the
#   predicted high up by about 1 degree for each degree of the low.
# - A surprising result: adding precipitation, wind and is_summer did not improve
#   the model at all, and precipitation and wind changed sign or shrank to near
#   zero compared to their raw correlations.