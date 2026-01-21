"""
Prediction generation module
Creates predictions and saves results
"""

import pandas as pd
import numpy as np

def generate_predictions(model, X, df, split_idx):
    """
    Generate predictions for entire dataset
    
    Args:
        model: Trained model
        X: Feature dataframe
        df: Complete dataframe with dates
        split_idx: Train/test split index
        
    Returns:
        predictions_df: Dataframe with dates, actual, and predicted prices
    """
    # Generate predictions
    predictions = np.round(model.predict(X), 2)
    
    # Create results dataframe
    predictions_df = pd.DataFrame({
        'Date': df['Date'],
        'Actual_Price': df['Target_Price'],
        'Predicted_Price': predictions,
        'Error': df['Target_Price'] - predictions,
        'Absolute_Error': np.abs(df['Target_Price'] - predictions),
        'Percentage_Error': np.abs((df['Target_Price'] - predictions) / df['Target_Price']) * 100,
        'Dataset': ['Train' if i < split_idx else 'Test' for i in range(len(df))]
    })
    
    return predictions_df