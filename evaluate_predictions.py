import pandas as pd
import numpy as np
import os

PRED_FILE = "outputs/predictions.csv"
OUT_FILE = "outputs/evaluation_summary.txt"

os.makedirs("outputs", exist_ok=True)

# Load predictions
df = pd.read_csv(PRED_FILE)

# Regression Metrics
mae = df["Absolute_Error"].mean()
rmse = np.sqrt(((df["Actual_Price"] - df["Predicted_Price"]) ** 2).mean())

r2 = 1 - (
    ((df["Actual_Price"] - df["Predicted_Price"]) ** 2).sum()
    / ((df["Actual_Price"] - df["Actual_Price"].mean()) ** 2).sum()
)

# Accuracy functions
def pct_accuracy(df, tol_pct):
    return (
        (np.abs(df["Actual_Price"] - df["Predicted_Price"])
         / df["Actual_Price"] * 100 <= tol_pct)
        .mean() * 100
    )

def price_accuracy(df, tol_price):
    return (
        np.abs(df["Actual_Price"] - df["Predicted_Price"]) <= tol_price
    ).mean() * 100

# Build report (TEXT ONLY)
report = f"""
PERCENTAGE-BASED ACCURACY
----------------------------------------
Accuracy within ±0.5% : {pct_accuracy(df, 0.5):.2f}%
Accuracy within ±1.0% : {pct_accuracy(df, 1.0):.2f}%
Accuracy within ±2.0% : {pct_accuracy(df, 2.0):.2f}%

PRICE-BASED ACCURACY
----------------------------------------
Accuracy within ±25  : {price_accuracy(df, 25):.2f}%
Accuracy within ±50  : {price_accuracy(df, 50):.2f}%
Accuracy within ±100 : {price_accuracy(df, 100):.2f}%

REGRESSION METRICS
----------------------------------------
MAE  : {mae:.2f}
RMSE : {rmse:.2f}
R²   : {r2:.4f}
"""

# Save file (UTF-8 safe)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(report.strip())

# Console confirmation
print("Evaluation saved successfully!")
print(f"File: {OUT_FILE}")
