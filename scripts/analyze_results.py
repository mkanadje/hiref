"""
Analyze model results by generating predictions and contributions for all data.

This script:
1. Loads the trained model and preprocessor
2. Runs predict_batch on train, validation, and test sets
3. Outputs comprehensive CSV with predictions, actuals, and driver contributions
"""

import sys

sys.path.append(".")

import pandas as pd
import numpy as np
from hier_reg.inference.predictor import Predictor
import config
import os


def analyze_results(
    data_path=config.DATA_PATH,
    model_path=config.MODEL_SAVE_PATH,
    preprocessor_path=config.PREPROCESSOR_SAVE_PATH,
    output_dir=config.RESULTS_PATH,
    device=config.DEVICE,
):
    """
    Generate comprehensive analysis CSV with predictions and contributions.

    Args:
        data_path: Path to the full dataset CSV
        model_path: Path to trained model weights
        preprocessor_path: Path to saved preprocessor
        output_dir: Directory to save output CSV
        device: Device to run inference on ('cpu', 'cuda', 'mps')

    Returns:
        pd.DataFrame: Complete results dataframe
    """

    print("=" * 80)
    print("MODEL RESULTS ANALYSIS")
    print("=" * 80)

    # ==========================================================================
    # STEP 1: Load data
    # ==========================================================================

    print("\n[1/5] Loading data...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    print(f"  Total records: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Split into train/val/test for labeling
    train_mask = df["date"] <= config.TRAIN_END_DATE
    val_mask = (df["date"] > config.TRAIN_END_DATE) & (
        df["date"] <= config.VAL_END_DATE
    )
    test_mask = df["date"] > config.VAL_END_DATE

    df["dataset_split"] = "unknown"
    df.loc[train_mask, "dataset_split"] = "train"
    df.loc[val_mask, "dataset_split"] = "val"
    df.loc[test_mask, "dataset_split"] = "test"

    print(f"  Train: {train_mask.sum():,} records")
    print(f"  Val:   {val_mask.sum():,} records")
    print(f"  Test:  {test_mask.sum():,} records")

    # ==========================================================================
    # STEP 2: Load model and preprocessor
    # ==========================================================================

    print(f"\n[2/5] Loading model and preprocessor...")
    predictor = Predictor(
        model_path=model_path, preprocessor_path=preprocessor_path, device=device
    )

    # ==========================================================================
    # STEP 3: Run batch prediction on all data
    # ==========================================================================

    print(f"\n[3/5] Running batch prediction on all {len(df):,} records...")

    # Prepare input data (all columns needed for prediction)
    input_cols = config.HIERARCHY_COLS + config.FEATURE_COLS + ["date"]
    input_df = df[input_cols].copy()

    # Run batch prediction (vectorized - fast!)
    results_df = predictor.predict_batch(input_df, return_dataframe=True)

    print(f"  Predictions generated: {len(results_df):,}")

    # ==========================================================================
    # STEP 4: Merge with original data
    # ==========================================================================

    print(f"\n[4/5] Merging results with original data...")

    # Add actual sales and dataset split to results
    results_df["actual_sales"] = df["sales"].values
    results_df["dataset_split"] = df["dataset_split"].values

    # Add all hierarchy columns
    for col in config.HIERARCHY_COLS:
        results_df[col] = df[col].values

    # Add all feature columns
    for col in config.FEATURE_COLS:
        results_df[col] = df[col].values

    # Calculate prediction error metrics
    results_df["error"] = results_df["prediction"] - results_df["actual_sales"]
    results_df["abs_error"] = np.abs(results_df["error"])
    results_df["pct_error"] = (results_df["error"] / results_df["actual_sales"]) * 100
    results_df["abs_pct_error"] = np.abs(results_df["pct_error"])

    # ==========================================================================
    # STEP 5: Prepare final output
    # ==========================================================================

    print(f"\n[5/5] Preparing final output...")

    # Note: Driver contributions are already flattened in the new vectorized predict_batch
    # No need to extract from dictionaries anymore!

    # Reorder columns for readability
    column_order = (
        [
            # Identifiers
            "date",
            "sku_key",
            "dataset_split",
        ]
        + config.HIERARCHY_COLS
        + [
            # Actuals and predictions
            "actual_sales",
            "prediction",
            "error",
            "abs_error",
            "pct_error",
            "abs_pct_error",
            # Contribution breakdown
            "baseline_original",
            "total_driver_contribution_original",
            "baseline_scaled",
            "total_driver_contribution_scaled",
            "bias",
            "embedding_contribution_scaled",
        ]
        + [f"driver_{f}_original" for f in config.FEATURE_COLS]
        + [f"driver_{f}_scaled" for f in config.FEATURE_COLS]
        + config.FEATURE_COLS
    )

    results_df = results_df[column_order]

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    output_path = os.path.join(output_dir, "model_results_with_contributions.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\n Results saved to: {output_path}")
    print(f"  Total records: {len(results_df):,}")
    print(f"  Total columns: {len(results_df.columns)}")

    # ==========================================================================
    # Print summary statistics
    # ==========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        split_df = results_df[results_df["dataset_split"] == split]
        if len(split_df) == 0:
            continue

        print(f"\n{split.upper()} SET ({len(split_df):,} records):")
        print(f"  MAE:  {split_df['abs_error'].mean():.2f}")
        print(f"  RMSE: {np.sqrt((split_df['error'] ** 2).mean()):.2f}")

        # MAPE excluding low sales
        mape_mask = split_df["actual_sales"] >= 50
        if mape_mask.sum() > 0:
            mape = split_df[mape_mask]["abs_pct_error"].mean()
            coverage = mape_mask.sum() / len(split_df) * 100
            print(f"  MAPE: {mape:.2f}% (coverage: {coverage:.1f}%)")

        # Average contributions
        print(
            f"  Avg baseline (original):      {split_df['baseline_original'].mean():.2f}"
        )
        print(
            f"  Avg driver contrib (original): {split_df['total_driver_contribution_original'].mean():.2f}"
        )

        # Calculate contribution percentages
        total_magnitude = (
            split_df["baseline_original"].abs()
            + split_df["total_driver_contribution_original"].abs()
        )
        baseline_pct = (
            split_df["baseline_original"].abs() / total_magnitude * 100
        ).mean()
        driver_pct = (
            split_df["total_driver_contribution_original"].abs() / total_magnitude * 100
        ).mean()

        print(f"  Avg baseline %:                {baseline_pct:.1f}%")
        print(f"  Avg driver %:                  {driver_pct:.1f}%")

    # Top drivers analysis
    print("\n" + "=" * 80)
    print("TOP DRIVER CONTRIBUTIONS (Original Scale, Test Set)")
    print("=" * 80)

    test_df = results_df[results_df["dataset_split"] == "test"]
    if len(test_df) > 0:
        print("\nAverage absolute contribution by driver:")
        driver_cols = [f"driver_{f}_original" for f in config.FEATURE_COLS]
        avg_abs_contrib = test_df[driver_cols].abs().mean().sort_values(ascending=False)

        for col, value in avg_abs_contrib.items():
            feature_name = col.replace("driver_", "").replace("_original", "")
            print(f"  {feature_name:25s}: {value:8.2f}")

    print("\n" + "=" * 80)

    return results_df


if __name__ == "__main__":
    results = analyze_results()
    print("\n Analysis complete!")
