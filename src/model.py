"""
Model definition module
Returns Linear Regression model
"""

from sklearn.linear_model import LinearRegression

def get_model():
    """
    Create and return Linear Regression model
    
    Returns:
        model: Scikit-learn LinearRegression instance
    """
    return LinearRegression()