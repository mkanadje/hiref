import pytest
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append(".")

from training.metrics import (
    calculate_metrics,
    mse,
    rmse,
    mae,
    mape,
    _to_numpy,
)


class TestMetricFunctions:
    """Test individual metric functions with numpy arrays."""

    def test_mse_perfect_prediction(self):
        """MSE should be 0 for perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert mse(predictions, targets) == 0.0

    def test_mse_calculation(self):
        """MSE should calculate correctly."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])
        # Errors: [-1, -1, -1], squared: [1, 1, 1], mean: 1.0
        assert mse(predictions, targets) == 1.0

    def test_rmse_calculation(self):
        """RMSE should be sqrt of MSE."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])
        # MSE = 1.0, RMSE = sqrt(1.0) = 1.0
        assert rmse(predictions, targets) == 1.0

    def test_rmse_larger_errors(self):
        """RMSE should calculate correctly for larger errors."""
        predictions = np.array([0.0, 0.0, 0.0])
        targets = np.array([3.0, 4.0, 5.0])
        # Errors: [-3, -4, -5], squared: [9, 16, 25], mean: 50/3, sqrt: ~4.08
        expected_rmse = np.sqrt((9 + 16 + 25) / 3)
        assert np.isclose(rmse(predictions, targets), expected_rmse)

    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert mae(predictions, targets) == 0.0

    def test_mae_calculation(self):
        """MAE should calculate correctly."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 4.0, 5.0])
        # Absolute errors: [1, 2, 2], mean: 5/3
        expected_mae = (1 + 2 + 2) / 3
        assert np.isclose(mae(predictions, targets), expected_mae)

    def test_mae_with_negative_errors(self):
        """MAE should handle negative errors correctly."""
        predictions = np.array([5.0, 3.0, 1.0])
        targets = np.array([2.0, 4.0, 5.0])
        # Errors: [3, -1, -4], absolute: [3, 1, 4], mean: 8/3
        expected_mae = (3 + 1 + 4) / 3
        assert np.isclose(mae(predictions, targets), expected_mae)


class TestMAPE:
    """Test MAPE function with various edge cases."""

    def test_mape_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        predictions = np.array([100.0, 200.0, 300.0])
        targets = np.array([100.0, 200.0, 300.0])
        mape_value, coverage = mape(predictions, targets)
        assert mape_value == 0.0
        assert coverage == 100.0

    def test_mape_calculation(self):
        """MAPE should calculate percentage errors correctly."""
        predictions = np.array([90.0, 110.0])
        targets = np.array([100.0, 100.0])
        # Percentage errors: |100-90|/100 = 10%, |100-110|/100 = 10%
        # Mean: 10%
        mape_value, coverage = mape(predictions, targets)
        assert np.isclose(mape_value, 10.0)
        assert coverage == 100.0

    def test_mape_with_zero_targets(self):
        """MAPE should skip zero targets and report coverage."""
        predictions = np.array([10.0, 20.0, 30.0, 40.0])
        targets = np.array([100.0, 0.0, 0.0, 200.0])
        # Only first and last are valid
        # Errors: |100-10|/100 = 90%, |200-40|/200 = 80%
        # Mean: 85%
        mape_value, coverage = mape(predictions, targets)
        assert np.isclose(mape_value, 85.0)
        assert coverage == 50.0  # 2 out of 4 samples

    def test_mape_all_zero_targets(self):
        """MAPE should return NaN when all targets are zero."""
        predictions = np.array([10.0, 20.0, 30.0])
        targets = np.array([0.0, 0.0, 0.0])
        mape_value, coverage = mape(predictions, targets)
        assert np.isnan(mape_value)
        assert coverage == 0.0

    def test_mape_with_small_epsilon(self):
        """MAPE should treat very small values as zero."""
        predictions = np.array([10.0, 20.0])
        targets = np.array([100.0, 1e-12])  # Second value below epsilon
        mape_value, coverage = mape(predictions, targets)
        # Only first value is valid: |100-10|/100 = 90%
        assert np.isclose(mape_value, 90.0)
        assert coverage == 50.0


class TestToNumpy:
    """Test conversion from PyTorch tensors to numpy arrays."""

    def test_torch_tensor_conversion(self):
        """Should convert PyTorch tensors to numpy arrays."""
        pred_tensor = torch.tensor([1.0, 2.0, 3.0])
        target_tensor = torch.tensor([4.0, 5.0, 6.0])

        preds, targets = _to_numpy(pred_tensor, target_tensor)

        assert isinstance(preds, np.ndarray)
        assert isinstance(targets, np.ndarray)
        np.testing.assert_array_equal(preds, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(targets, np.array([4.0, 5.0, 6.0]))

    def test_numpy_array_passthrough(self):
        """Should handle numpy arrays without conversion."""
        pred_array = np.array([1.0, 2.0, 3.0])
        target_array = np.array([4.0, 5.0, 6.0])

        preds, targets = _to_numpy(pred_array, target_array)

        assert isinstance(preds, np.ndarray)
        assert isinstance(targets, np.ndarray)
        np.testing.assert_array_equal(preds, pred_array)
        np.testing.assert_array_equal(targets, target_array)

    def test_mixed_types(self):
        """Should handle mixed PyTorch and numpy inputs."""
        pred_tensor = torch.tensor([1.0, 2.0, 3.0])
        target_array = np.array([4.0, 5.0, 6.0])

        preds, targets = _to_numpy(pred_tensor, target_array)

        assert isinstance(preds, np.ndarray)
        assert isinstance(targets, np.ndarray)

    def test_gradient_detachment(self):
        """Should detach tensors with gradients."""
        pred_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target_tensor = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

        preds, targets = _to_numpy(pred_tensor, target_tensor)

        # Should not raise error about gradients
        assert isinstance(preds, np.ndarray)
        assert isinstance(targets, np.ndarray)


class TestCalculateMetrics:
    """Test the main calculate_metrics function."""

    def test_calculate_metrics_with_numpy(self):
        """Should calculate all metrics correctly with numpy arrays."""
        predictions = np.array([90.0, 110.0, 95.0, 105.0])
        targets = np.array([100.0, 100.0, 100.0, 100.0])

        metrics = calculate_metrics(predictions, targets)

        # Check all expected keys are present
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "mape_coverage" in metrics
        assert "r2" in metrics

        # Check MSE: errors = [-10, 10, -5, 5], squared = [100, 100, 25, 25], mean = 62.5
        assert np.isclose(metrics["mse"], 62.5)

        # Check RMSE: sqrt(62.5) ≈ 7.906
        assert np.isclose(metrics["rmse"], np.sqrt(62.5))

        # Check MAE: |errors| = [10, 10, 5, 5], mean = 7.5
        assert np.isclose(metrics["mae"], 7.5)

        # Check MAPE: [10%, 10%, 5%, 5%], mean = 7.5%
        assert np.isclose(metrics["mape"], 7.5)

        # Check coverage: all samples valid
        assert metrics["mape_coverage"] == 100.0

    def test_calculate_metrics_with_torch(self):
        """Should calculate all metrics correctly with PyTorch tensors."""
        predictions = torch.tensor([90.0, 110.0, 95.0, 105.0])
        targets = torch.tensor([100.0, 100.0, 100.0, 100.0])

        metrics = calculate_metrics(predictions, targets)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "r2" in metrics

        # Same calculations as numpy test
        assert np.isclose(metrics["mse"], 62.5)
        assert np.isclose(metrics["mae"], 7.5)

    def test_calculate_metrics_with_scaler(self):
        """Should inverse transform predictions and targets when scaler is provided."""
        # Create scaled data
        original_targets = np.array([100.0, 200.0, 300.0, 400.0]).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(original_targets)

        scaled_targets = scaler.transform(original_targets).flatten()
        # Perfect predictions in scaled space
        scaled_predictions = scaled_targets.copy()

        metrics = calculate_metrics(scaled_predictions, scaled_targets, scaler=scaler)

        # Metrics should be calculated in original scale
        # Perfect predictions should give MSE = 0
        assert np.isclose(metrics["mse"], 0.0)
        assert np.isclose(metrics["rmse"], 0.0)
        assert np.isclose(metrics["mae"], 0.0)
        assert np.isclose(metrics["mape"], 0.0)

    def test_calculate_metrics_r2_score(self):
        """Should calculate R2 score correctly."""
        predictions = np.array([3.0, -0.5, 2.0, 7.0])
        targets = np.array([2.5, 0.0, 2.0, 8.0])

        metrics = calculate_metrics(predictions, targets)

        # R2 should be between -inf and 1
        # For good predictions, should be close to 1
        assert "r2" in metrics
        assert metrics["r2"] <= 1.0

    def test_calculate_metrics_with_zero_targets(self):
        """Should handle zero targets in MAPE calculation."""
        predictions = np.array([10.0, 20.0, 30.0, 40.0])
        targets = np.array([100.0, 0.0, 0.0, 200.0])

        metrics = calculate_metrics(predictions, targets)

        # MAPE should skip zeros
        assert metrics["mape_coverage"] == 50.0
        assert not np.isnan(metrics["mape"])

        # Other metrics should still work
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arrays(self):
        """Should handle empty arrays gracefully."""
        predictions = np.array([])
        targets = np.array([])

        # MSE of empty array should be NaN
        result = mse(predictions, targets)
        assert np.isnan(result)

    def test_single_value(self):
        """Should handle single value predictions."""
        predictions = np.array([5.0])
        targets = np.array([10.0])

        assert mse(predictions, targets) == 25.0
        assert rmse(predictions, targets) == 5.0
        assert mae(predictions, targets) == 5.0

    def test_large_values(self):
        """Should handle large values without overflow."""
        predictions = np.array([1e6, 2e6, 3e6])
        targets = np.array([1.1e6, 2.1e6, 3.1e6])

        metrics = calculate_metrics(predictions, targets)

        # Should not overflow or return inf
        assert np.isfinite(metrics["mse"])
        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["mae"])

    def test_negative_values(self):
        """Should handle negative values correctly."""
        predictions = np.array([-10.0, -20.0, -30.0])
        targets = np.array([-15.0, -25.0, -35.0])

        metrics = calculate_metrics(predictions, targets)

        # MSE: errors = [5, 5, 5], squared = [25, 25, 25], mean = 25
        assert np.isclose(metrics["mse"], 25.0)
        assert np.isclose(metrics["mae"], 5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
