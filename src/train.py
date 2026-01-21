"""
Model training module
Handles train-test split and model evaluation
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(model, X, y):
    """
    Train model with time-based split
    
    Args:
        model: Scikit-learn model instance
        X: Feature dataframe
        y: Target series
        
    Returns:
        model: Trained model
        metrics: Dictionary of performance metrics
        split_idx: Index where train/test split occurred
    """
    # Time-based split (80-20)
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
    }
    
    return model, metrics, split_idx