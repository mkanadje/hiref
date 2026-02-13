import torch
import numpy as np
from sklearn.metrics import r2_score


def calculate_metrics(predictions, targets, scaler=None):
    """
    Calculate regression metrics in original scale.

    Args:
        predictions: torch.Tensor or np.ndarray
            Model predictions (can be scaled or original)
        targets: torch.Tensor or np.ndarray
            Ground truth values (can be scaled or original)
        scaler: sklearn.preprocessing.StandardScaler, this is optional
            If provided, inverse tranform predictions and targets to original scale
            before computing the metrics.

    Returns:
        dict: Dictionary containing:
            - "mse": Mean Squared Error
            - "rmse": Root Mean Squared Error
            - "mae": Mean Absolute Error
            - "mape": Mean Absolute Percentage Error (%)
            - "r2": R-squared score
    """
    preds, targs = _to_numpy(predictions, targets)
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targs = scaler.inverse_transform(targs.reshape(-1, 1)).flatten()
    mape_value, mape_coverage = mape(preds, targs)
    return {
        "mse": mse(preds, targs),
        "rmse": rmse(preds, targs),
        "mae": mae(preds, targs),
        "mape": mape_value,
        "mape_coverage": mape_coverage,
        "r2": r2_score(preds, targs),
    }


def _to_numpy(predictions, targets):
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy()
    else:
        preds = np.asarray(predictions)
    if isinstance(targets, torch.Tensor):
        targs = targets.detach().cpu().numpy()
    else:
        targs = np.asarray(targets)
    return preds, targs


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def rmse(predictions, targets):
    return np.sqrt(mse(predictions, targets))


def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))


def mape(predictions, targets, min_threshold=50.0):
    """
    Calculate MAPE while excluding samples with very low actual values.

    Args:
        predictions: Predicted values
        targets: Actual values
        min_threshold: Minimum absolute value of target to include in MAPE calculation.
                      Samples with |target| < min_threshold are excluded to avoid
                      unrealistic percentage errors from near-zero denominators.
                      Default: 50.0 (reasonable for sales data in units)

    Returns:
        tuple: (mape_value, coverage_percentage)
            - mape_value: MAPE in percentage
            - coverage_percentage: % of samples used in calculation
    """
    # Only include samples where target is above threshold
    mask = np.abs(targets) >= min_threshold
    n_total = len(targets)
    n_valid = mask.sum()

    if n_valid == 0:
        return np.nan, 0.0  # All targets below threshold

    percentage_errors = np.abs((targets[mask] - predictions[mask]) / targets[mask])
    mape_value = np.mean(percentage_errors) * 100
    coverage = (n_valid / n_total) * 100

    return mape_value, coverage
