"""
Main pipeline execution script
Runs complete workflow: load → train → predict → evaluate
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_processed_data
from src.model import get_model
from src.train import train_model
from src.predict import generate_predictions


def main():
    print("=" * 80)
    print("STOCK PRICE PREDICTION - COMPLETE PIPELINE")
    print("=" * 80)

    # LOAD DATA
    X, y, df = load_processed_data()
    print(f"\nLoaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # MODEL
    model = get_model()
    print("Model: Linear Regression")

    # TRAIN
    model, metrics, split_idx = train_model(model, X, y)

    print("\nTRAIN METRICS")
    print(f"MAE  : {metrics['train']['mae']:.2f}")
    print(f"RMSE : {metrics['train']['rmse']:.2f}")
    print(f"R²   : {metrics['train']['r2']:.4f}")

    print("\nTEST METRICS")
    print(f"MAE  : {metrics['test']['mae']:.2f}")
    print(f"RMSE : {metrics['test']['rmse']:.2f}")
    print(f"R²   : {metrics['test']['r2']:.4f}")

    # PREDICTIONS
    predictions_df = generate_predictions(model, X, df, split_idx)

    os.makedirs("outputs", exist_ok=True)
    predictions_df.to_csv("outputs/predictions.csv", index=False)

    # COEFFICIENTS (FIXED ALIGNMENT)
    print("\nMODEL COEFFICIENTS")
    print("-" * 60)
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature:<25} : {coef:>12.6f}")
    print(f"{'Intercept':<25} : {model.intercept_:>12.6f}")
    print("-" * 60)

    # SAVE REPORT (CLEAN FORMAT)
    report = f"""
STOCK PRICE PREDICTION - MODEL REPORT
====================================

Train Samples : {split_idx}
Test Samples  : {len(X) - split_idx}

TRAIN METRICS
-------------
MAE  : {metrics['train']['mae']:.2f}
RMSE : {metrics['train']['rmse']:.2f}
R2   : {metrics['train']['r2']:.2f}

TEST METRICS
------------
MAE  : {metrics['test']['mae']:.2f}
RMSE : {metrics['test']['rmse']:.2f}
R2   : {metrics['test']['r2']:.2f}

MODEL COEFFICIENTS
------------------
Price_Lag1        : {model.coef_[0]:.2f}
Data_Lag1         : {model.coef_[1]:.2f}
Data_Change_Lag1  : {model.coef_[2]:.2f}
Intercept         : {model.intercept_:.2f}

"""

    with open("outputs/model_performance.txt", "w") as f:
        f.write(report)

    print("\nPipeline completed successfully.")
    print("Outputs saved in /outputs")


if __name__ == "__main__":
    main()
