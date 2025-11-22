"""
Manual test script for the Predictor class.
This script demonstrates how to use the predictor for single SKU predictions.
"""

import sys

sys.path.append(".")

from inference.predictor import Predictor
import json

# =============================================================================
# INITIALIZE PREDICTOR
# =============================================================================

print("=" * 70)
print("TESTING PREDICTOR")
print("=" * 70)

# Paths to trained model and preprocessor
model_path = "./outputs/model.pt"
preprocessor_path = "./outputs/preprocessor.pkl"

print("\nInitializing Predictor...")
print(f"Model path: {model_path}")
print(f"Preprocessor path: {preprocessor_path}")

predictor = Predictor(
    model_path=model_path, preprocessor_path=preprocessor_path, device="mps"
)

print("\n" + "=" * 70)

# =============================================================================
# TEST CASE 1: Premium Brand in Northeast
# =============================================================================

print("\nTEST CASE 1: Premium Brand in Northeast")
print("-" * 70)

sku_dict_1 = {
    # Hierarchy
    "region": "Northeast",
    "state": "Northfield",
    "segment": "Premium",
    "brand": "Alpine Springs",
    "pack": "8oz",
    # Weather features
    "temperature": 72.5,
    "precipitation": 0.5,
    # Macroeconomic features
    "gdp_index": 105.2,
    "unemployment_rate": 3.5,
    "consumer_confidence": 98.5,
    # SKU-level features
    "price": 4.99,
    "distribution": 85.0,
    # Promotional features
    "tv_spend": 100.0,
    "digital_spend": 50.0,
    "trade_spend": 30.0,
    "discount_pct": 10.0,
}

print("\nInput SKU:")
for key, value in sku_dict_1.items():
    print(f"  {key:25s}: {value}")

result_1 = predictor.predict_single(sku_dict_1)

print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

print(f"\nSKU Key: {result_1['sku_key']}")
print(f"\nPredicted Sales (Original Scale): {result_1['prediction']:.2f} units")

print("\n" + "-" * 70)
print("DECOMPOSITION (Original Scale - For Business)")
print("-" * 70)
print(f"Baseline (inherent SKU potential): {result_1['baseline_original']:.2f}")
print(
    f"Total Driver Impact:                {result_1['total_driver_contribution_original']:.2f}"
)
print(f"  = Prediction:                     {result_1['prediction']:.2f}")

print("\nDriver Contributions (Original Scale):")
for feature, contrib in sorted(
    result_1["driver_contribution_original"].items(),
    key=lambda x: abs(x[1]),
    reverse=True,
):
    sign = "+" if contrib >= 0 else ""
    print(f"  {feature:25s}: {sign}{contrib:8.2f}")

print("\n" + "-" * 70)
print("TECHNICAL DETAILS (Scaled Space - For Debugging)")
print("-" * 70)
print(f"Baseline (scaled):        {result_1['baseline_scaled']:.4f}")
print(f"  = Bias:                 {result_1['bias']:.4f}")
print(f"  + Embedding contrib:    {result_1['embedding_contribution_scaled']:.4f}")
print(f"Total Driver (scaled):    {result_1['total_driver_contribution_scaled']:.4f}")

print("\n" + "-" * 70)
print("VERIFICATION")
print("-" * 70)
# Verify additivity in original space
reconstructed = (
    result_1["baseline_original"] + result_1["total_driver_contribution_original"]
)
diff = abs(reconstructed - result_1["prediction"])
print(f"Baseline + Drivers = {reconstructed:.4f}")
print(f"Prediction         = {result_1['prediction']:.4f}")
print(f"Difference         = {diff:.10f} (should be ~0)")
print(f"Additivity Check:    {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")

# =============================================================================
# TEST CASE 2: Economy Brand in Southeast
# =============================================================================

print("\n\n" + "=" * 70)
print("TEST CASE 2: Economy Brand in Southeast")
print("-" * 70)

sku_dict_2 = {
    # Hierarchy
    "region": "Southeast",
    "state": "Suncoast",
    "segment": "Economy",
    "brand": "Value Plus",
    "pack": "16oz",
    # Weather features
    "temperature": 85.0,
    "precipitation": 0.2,
    # Macroeconomic features
    "gdp_index": 103.5,
    "unemployment_rate": 4.5,
    "consumer_confidence": 95.0,
    # SKU-level features
    "price": 2.99,
    "distribution": 92.0,
    # Promotional features
    "tv_spend": 30.0,
    "digital_spend": 20.0,
    "trade_spend": 40.0,
    "discount_pct": 15.0,
}

print("\nInput SKU:")
for key, value in sku_dict_2.items():
    print(f"  {key:25s}: {value}")

result_2 = predictor.predict_single(sku_dict_2)

print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

print(f"\nSKU Key: {result_2['sku_key']}")
print(f"\nPredicted Sales (Original Scale): {result_2['prediction']:.2f} units")

print("\n" + "-" * 70)
print("DECOMPOSITION (Original Scale - For Business)")
print("-" * 70)
print(f"Baseline (inherent SKU potential): {result_2['baseline_original']:.2f}")
print(
    f"Total Driver Impact:                {result_2['total_driver_contribution_original']:.2f}"
)
print(f"  = Prediction:                     {result_2['prediction']:.2f}")

print("\nDriver Contributions (Original Scale):")
for feature, contrib in sorted(
    result_2["driver_contribution_original"].items(),
    key=lambda x: abs(x[1]),
    reverse=True,
):
    sign = "+" if contrib >= 0 else ""
    print(f"  {feature:25s}: {sign}{contrib:8.2f}")

# =============================================================================
# TEST CASE 3: Same SKU, Different Drivers
# =============================================================================

print("\n\n" + "=" * 70)
print("TEST CASE 3: Same SKU as Test 1, but with Higher Promotion")
print("-" * 70)

sku_dict_3 = sku_dict_1.copy()
sku_dict_3["tv_spend"] = 200.0  # Double TV spend
sku_dict_3["digital_spend"] = 100.0  # Double digital spend
sku_dict_3["discount_pct"] = 20.0  # Higher discount

print("\nChanges from Test 1:")
print(f"  tv_spend:     {sku_dict_1['tv_spend']} -> {sku_dict_3['tv_spend']}")
print(
    f"  digital_spend: {sku_dict_1['digital_spend']} -> {sku_dict_3['digital_spend']}"
)
print(f"  discount_pct:  {sku_dict_1['discount_pct']} -> {sku_dict_3['discount_pct']}")

result_3 = predictor.predict_single(sku_dict_3)

print("\n" + "=" * 70)
print("COMPARISON: Test 1 vs Test 3")
print("=" * 70)
print(f"\nPrediction Test 1: {result_1['prediction']:.2f} units")
print(f"Prediction Test 3: {result_3['prediction']:.2f} units")
print(
    f"Lift:              {result_3['prediction'] - result_1['prediction']:.2f} units ({((result_3['prediction'] - result_1['prediction']) / result_1['prediction'] * 100):.1f}%)"
)

print(f"\nBaseline Test 1:   {result_1['baseline_original']:.2f} (should be same)")
print(f"Baseline Test 3:   {result_3['baseline_original']:.2f} (should be same)")

print("\nDriver Contributions Comparison:")
print(f"{'Feature':<25s} {'Test 1':>12s} {'Test 3':>12s} {'Change':>12s}")
print("-" * 70)
for feature in result_1["driver_contribution_original"].keys():
    contrib_1 = result_1["driver_contribution_original"][feature]
    contrib_3 = result_3["driver_contribution_original"][feature]
    change = contrib_3 - contrib_1
    print(f"{feature:<25s} {contrib_1:>12.2f} {contrib_3:>12.2f} {change:>12.2f}")

# =============================================================================
# EXPORT RESULTS TO JSON
# =============================================================================

print("\n\n" + "=" * 70)
print("EXPORTING RESULTS")
print("=" * 70)

output_path = "./outputs/predictor_test_results.json"
results = {
    "test_case_1": {
        "description": "Premium Brand in Northeast",
        "input": sku_dict_1,
        "output": result_1,
    },
    "test_case_2": {
        "description": "Economy Brand in Southeast",
        "input": sku_dict_2,
        "output": result_2,
    },
    "test_case_3": {
        "description": "Same SKU as Test 1 with Higher Promotion",
        "input": sku_dict_3,
        "output": result_3,
    },
}

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
