"""
Data preprocessing module
Loads processed data and prepares features and target
"""

import pandas as pd

def load_processed_data(path="data/processed/processed_data.csv"):
    """
    Load preprocessed data with lag features
    
    Args:
        path: Path to processed CSV file
        
    Returns:
        X: Feature dataframe
        y: Target series
        df: Complete dataframe
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    
    # Define features (lag-based approach)
    X = df[[
        "Price_Lag1",        # Yesterday's price 
        "Data_Lag1",         # Yesterday's data level
        "Data_Change_Lag1"   # Yesterday's data change 
    ]]
    
    # Target: Next day's price
    y = df["Target_Price"]
    
    return X, y, df