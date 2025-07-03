import numpy as np


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error between predicted and actual values

    Parameters:
    y_true (array-like): Ground truth (correct) target values
    y_pred (array-like): Estimated target values

    Returns:
    float: Root Mean Squared Error
    """
    # Convert inputs to numpy arrays for consistent handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate squared error for each prediction
    squared_errors = (y_true - y_pred) ** 2

    # Calculate mean of squared errors
    mean_squared_error = np.mean(squared_errors)

    # Take square root to get RMSE
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return root_mean_squared_error
