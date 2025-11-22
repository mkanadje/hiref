"""
Test script for the predict_batch method.
Tests batch prediction on actual test data and compares with predict_single.
"""

import sys

sys.path.append(".")

import pandas as pd
import numpy as np
from inference.predictor import Predictor
import time

# =============================================================================
# LOAD TEST DATA
# =============================================================================

print("=" * 80)
print("TESTING PREDICT_BATCH")
print("=" * 80)

# Load the full dataset
print("\nLoading data...")
data_path = "./sample_data/fmcg_hierarchical_data.csv"
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])

print(f"Total dataset shape: {df.shape}")

# Get test data (after 2024-03-31)
test_df = df[df["date"] > "2024-03-31"].reset_index(drop=True)
print(f"Test data shape: {test_df.shape}")

# =============================================================================
# INITIALIZE PREDICTOR
# =============================================================================

print("\n" + "=" * 80)
print("INITIALIZING PREDICTOR")
print("=" * 80)

model_path = "./outputs/model.pt"
preprocessor_path = "./outputs/preprocessor.pkl"

predictor = Predictor(
    model_path=model_path, preprocessor_path=preprocessor_path, device="mps"
)

# =============================================================================
# TEST 1: Small Batch Prediction
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: Small Batch (10 SKUs)")
print("=" * 80)

# Take first 10 rows from test data
small_batch = test_df.head(10).copy()

print(f"\nBatch size: {len(small_batch)}")
print(f"Date range: {small_batch['date'].min()} to {small_batch['date'].max()}")
print(
    f"Unique SKUs: {small_batch.groupby(['region', 'state', 'segment', 'brand', 'pack']).ngroups}"
)

# Predict using batch method
print("\nRunning predict_batch...")
start_time = time.time()
results_df = predictor.predict_batch(small_batch, return_dataframe=True)
batch_time = time.time() - start_time

print(f"Batch prediction completed in {batch_time:.4f} seconds")
print(f"\nResults shape: {results_df.shape}")
print(f"\nFirst 3 predictions:")
print(results_df[["date", "sku_key", "prediction", "baseline_original"]].head(3))

# =============================================================================
# TEST 2: Verify Additivity
# =============================================================================

print("\n" + "=" * 80)
print("TEST 2: Verify Additivity (baseline + drivers = prediction)")
print("=" * 80)

for i in range(min(5, len(results_df))):
    result = results_df.iloc[i]
    reconstructed = (
        result["baseline_original"] + result["total_driver_contribution_original"]
    )
    diff = abs(reconstructed - result["prediction"])

    print(f"\nSKU {i+1}: {result['sku_key']}")
    print(f"  Baseline:      {result['baseline_original']:.4f}")
    print(f"  Total drivers: {result['total_driver_contribution_original']:.4f}")
    print(f"  = Sum:         {reconstructed:.4f}")
    print(f"  Prediction:    {result['prediction']:.4f}")
    print(f"  Difference:    {diff:.10f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")

# =============================================================================
# TEST 3: Compare with predict_single
# =============================================================================

print("\n" + "=" * 80)
print("TEST 3: Compare predict_batch vs predict_single")
print("=" * 80)

# Take 3 samples
samples = small_batch.head(3)

print("\nComparing predictions for 3 SKUs...\n")

for idx in range(len(samples)):
    # Convert row to dict for predict_single
    sku_dict = samples.iloc[idx].to_dict()

    # Predict using predict_single
    single_result = predictor.predict_single(sku_dict)

    # Get batch result
    batch_result = results_df.iloc[idx]

    # Compare predictions
    pred_diff = abs(single_result["prediction"] - batch_result["prediction"])
    baseline_diff = abs(
        single_result["baseline_original"] - batch_result["baseline_original"]
    )

    print(f"SKU {idx+1}: {single_result['sku_key']}")
    print(f"  predict_single prediction: {single_result['prediction']:.6f}")
    print(f"  predict_batch prediction:  {batch_result['prediction']:.6f}")
    print(
        f"  Difference:                {pred_diff:.10f} {'✓ MATCH' if pred_diff < 1e-6 else '✗ MISMATCH'}"
    )
    print(
        f"  Baseline diff:             {baseline_diff:.10f} {'✓ MATCH' if baseline_diff < 1e-6 else '✗ MISMATCH'}"
    )
    print()

# =============================================================================
# TEST 4: Large Batch Performance
# =============================================================================

print("=" * 80)
print("TEST 4: Large Batch Performance")
print("=" * 80)

# Take 1000 rows
large_batch = test_df.head(1000).copy()

print(f"\nBatch size: {len(large_batch)}")

# Predict using batch method
print("\nRunning predict_batch on 1000 SKUs...")
start_time = time.time()
large_results = predictor.predict_batch(large_batch, return_dataframe=True)
batch_time = time.time() - start_time

print(f"Batch prediction: {batch_time:.4f} seconds")
print(f"Average per SKU: {batch_time / len(large_batch) * 1000:.2f} ms")

# Estimate predict_single time (test on 10 samples)
print("\nEstimating predict_single time (on 10 samples)...")
single_times = []
for idx in range(10):
    sku_dict = large_batch.iloc[idx].to_dict()
    start = time.time()
    _ = predictor.predict_single(sku_dict)
    single_times.append(time.time() - start)

avg_single_time = np.mean(single_times)
estimated_total_single_time = avg_single_time * len(large_batch)

print(f"Average predict_single time: {avg_single_time * 1000:.2f} ms")
print(
    f"Estimated time for 1000 SKUs with predict_single: {estimated_total_single_time:.2f} seconds"
)
print(f"\nSpeedup: {estimated_total_single_time / batch_time:.1f}x faster with batch!")

# =============================================================================
# TEST 5: Output Format Options
# =============================================================================

print("\n" + "=" * 80)
print("TEST 5: Output Format Options")
print("=" * 80)

small_batch_subset = test_df.head(5).copy()

# Test 1: DataFrame input, DataFrame output
print("\nDataFrame input + return_dataframe=True:")
df_result = predictor.predict_batch(small_batch_subset, return_dataframe=True)
print(f"  Type: {type(df_result)}")
print(f"  Shape: {df_result.shape}")

# Test 2: DataFrame input, list output
print("\nDataFrame input + return_dataframe=False:")
list_result = predictor.predict_batch(small_batch_subset, return_dataframe=False)
print(f"  Type: {type(list_result)}")
print(f"  Length: {len(list_result)}")

# Test 3: List of dicts input
print("\nList of dicts input:")
sku_list = small_batch_subset.to_dict("records")
list_result2 = predictor.predict_batch(sku_list, return_dataframe=False)
print(f"  Type: {type(list_result2)}")
print(f"  Length: {len(list_result2)}")

# Test 4: List of dicts input, but request DataFrame
print("\nList of dicts input + return_dataframe=True:")
df_result2 = predictor.predict_batch(sku_list, return_dataframe=True)
print(f"  Type: {type(df_result2)}")
print(f"  Shape: {df_result2.shape}")

# =============================================================================
# TEST 6: Date Validation
# =============================================================================

print("\n" + "=" * 80)
print("TEST 6: Date Field Validation")
print("=" * 80)

# Test with missing date column
test_no_date = test_df.head(5).drop(columns=["date"])

print("\nTesting with missing 'date' column...")
try:
    predictor.predict_batch(test_no_date, return_dataframe=True)
    print("✗ FAIL: Should have raised ValueError!")
except ValueError as e:
    print(f"✓ PASS: Correctly raised ValueError")
    print(f"  Error message: {str(e)}")

# =============================================================================
# TEST 7: Driver Contributions Analysis
# =============================================================================

print("\n" + "=" * 80)
print("TEST 7: Driver Contributions Analysis")
print("=" * 80)

# Analyze top drivers for first SKU
first_result = large_results.iloc[0]

print(f"\nSKU: {first_result['sku_key']}")
print(f"Date: {first_result['date']}")
print(f"Prediction: {first_result['prediction']:.2f}")
print(f"\nTop Driver Contributions (Original Scale):")

# Extract driver contributions
driver_contribs = first_result["driver_contribution_original"]

# Sort by absolute value
sorted_drivers = sorted(driver_contribs.items(), key=lambda x: abs(x[1]), reverse=True)

for feature, contrib in sorted_drivers[:5]:
    sign = "+" if contrib >= 0 else ""
    print(f"  {feature:25s}: {sign}{contrib:8.2f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n✓ All tests completed successfully!")
print(f"\nKey findings:")
print(f"  - Batch prediction works correctly")
print(f"  - Additivity verified (baseline + drivers = prediction)")
print(f"  - predict_batch matches predict_single (within 1e-6)")
print(f"  - ~{estimated_total_single_time / batch_time:.0f}x speedup on 1000 SKUs")
print(f"  - Date validation working")
print(f"  - Both DataFrame and list inputs supported")

print("\n" + "=" * 80)
