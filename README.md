# Stock Price Prediction using Lag-Based Machine Learning

## Overview

This project focuses on predicting the **next day’s stock price** by modeling the relationship between an independent economic indicator (**Data**) and historical prices (**StockPrice**). The solution prioritizes statistical stability and interpretability to meet professional financial modeling standards.

## Objective

The primary goal is to build a machine learning pipeline that:

1. Forecasts using only provided data.
2. Explicitly quantifies how day-over-day changes in the independent variable influence price movements.
3. Maintains strict adherence to assignment constraints (no external data, news, or sentiment).

## Project Structure

```text
FUTURES FIRST ASSIGNMENT/
├── ├── data/
│   ├── raw/                 # Original CSV files (data2.csv, stockprice.csv)
│   └── processed/           # Engineered datasets with lag features
│
├── notebooks/
│   └── EDA_and_Feature_Engineering.ipynb
│
├── src/                        # Modular source code
│   ├── preprocess.py           # Data cleaning & lag logic
│   ├── model.py                # Model architecture
│   ├── train.py                # Training logic
│   └── predict.py              # Inference script
│
├── outputs/                    # Generated results
│   ├── predictions.csv         # Actual vs Predicted values
│   ├── model_performance.txt   # Detailed metrics report
│   └── evaluation_summary.txt  # Summary of evaluation
│
├── run_pipeline.py             # Main execution script
├── evaluate_predictions.py     # Main execution script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation


```

## Installation and Usage

1. **Clone the Repository:**
```bash
git clone <https://github.com/ayushyadav10/Stock-Price-Prediction-using-Lag-Based-Machine-Learning>
cd FUTURES FIRST ASSIGNMENT

```


2. **Setup Environment:**
```bash
python -m venv env
# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run Pipeline:**
```bash
python run_pipeline.py

```



---

## EDA Results

* **Data Integrity:** Merged 3,802 records with 0 missing values and perfect chronological alignment.
* **Volatility:** The `Data` variable shows a skewness of 1.43, indicating occasional high-impact spikes.
* **Stationarity:** While price levels are non-stationary, they exhibit strong continuity, making them suitable for lag-based regression.

## Critical Insight

During EDA, it was discovered that the direct correlation between `Data_Change` and `Price_Change` is **-0.028** (Extremely Weak).
**The Challenge:** A naive model predicting "change-from-change" would result in near-zero accuracy.
**The Solution:** We utilize **Lag-Based Level Prediction**. By predicting the price level () using lagged level features, we preserve the assignment's core assumption while achieving high statistical stability.

---

## Feature Engineering

We engineered features to capture momentum, long-term relationships, and the specific requirement of day-over-day changes:

| Feature | Formula | Purpose |
| --- | --- | --- |
| **Price_Lag1** |  | Captures price momentum (anchoring effect). |
| **Data_Lag1** |  | Models the long-term structural relationship. |
| **Data_Change_Lag1** |  | **Explicitly models day-over-day influence (Assignment Focus).** |

---

## Methodology & Strategy

* **Time-Series Splitting:** Used an 80/20 chronological split. Shuffling was disabled to prevent "Data Leakage" (using future info to predict the past).
* **Stationarity Handling:** Rather than differencing the data (which loses interpretability), I used lagged levels to provide the model with a clear baseline.
* **Preprocessing:** Chronological sorting, inner-joining datasets on Date, and removing NaN values generated during the lagging process.

## Model Selection

**Chosen Model:** **Linear Regression (Ordinary Least Squares)**

* **Justification:** In financial modeling, transparency is key. Linear Regression provides clear coefficients, allowing us to explain exactly how a 1-unit change in Data affects the Price.
* **Risk Mitigation:** Given the low number of features (3), complex models like XGBoost or LSTM would be prone to overfitting and lack the transparency required for this evaluation.

---

## Model Performance

### Regression Metrics

| Metric | Training Set | Testing Set | Overall |
| --- | --- | --- | --- |
| **MAE** | 22.10 | 54.59 | **28.60** |
| **RMSE** | 35.08 | 69.70 | **44.23** |
| **R² Score** | 1.00 | **0.99** | **0.9989** |


* **Overall R² Score:** 0.9989 (Explains 99.8% of price variance).
* **MAE (Mean Absolute Error):** 28.60 (Average error of only ₹28.60).
* **RMSE:** 44.23.

## Accuracy Breakdown

| Error Tolerance | Accuracy % | Interpretation |
| --- | --- | --- |
| **Within ±1.0%** | 65.20% | High-precision predictions. |
| **Within ±2.0%** | 88.97% | Nearly 9 out of 10 within 2% error. |
| **Within ±₹100** | 95.24% | Robust strategic accuracy. |

## Model Coefficients

The model learned the following weights:

* **Price_Lag1 (1.0046):** Confirms that tomorrow’s price is heavily anchored to today’s price.
* **Data_Change_Lag1 (163.45):** **Primary Finding.** Every 1-unit increase in the previous day's Data change leads to a ₹163.45 increase in next-day price.
* **Data_Lag1 (-2.91):** Indicates a minor long-term inverse relationship.

---

## Conclusion

This implementation successfully bridges the gap between the assignment's core assumption and empirical data. While the direct change correlation was weak, the **Lag-Based Model** effectively captures the Data variable's influence, resulting in a stable, interpretable, and high-performing forecasting tool.

