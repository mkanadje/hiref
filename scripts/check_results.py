import pandas as pd
import numpy as np

try:
    df = pd.read_csv("./outputs/results/model_results_with_contributions.csv")
    print("Columns:", df.columns)
    print("\nChecking for NaNs:")
    print(df[["prediction", "actual_sales", "dataset_split"]].isna().sum())

    print("\nChecking for Infs:")
    print(np.isinf(df[["prediction", "actual_sales"]]).sum())

    print("\nSample data:")
    print(df[["prediction", "actual_sales", "dataset_split"]].head())

    # Check for non-numeric types
    print("\nData Types:")
    print(df[["prediction", "actual_sales"]].dtypes)

except Exception as e:
    print(f"Error reading file: {e}")
