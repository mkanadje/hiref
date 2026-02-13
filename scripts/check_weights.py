import sys

sys.path.append(".")

import torch
import pickle
from models.hierarchical_model import HierarchicalModel
import config

# Load preprocessor
with open("./outputs/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Get constraint info
constraint_indices = preprocessor["constraint_indices"]
print("Constraint indices from preprocessor:")
print(constraint_indices)
print()

# Create model architecture
vocab_sizes = preprocessor["vocab_sizes"]
model = HierarchicalModel(
    vocab_sizes=vocab_sizes,
    embedding_dims=config.EMBEDDING_DIMS,
    n_features=len(config.FEATURE_COLS),
    constraint_indices=constraint_indices,
    projection_dim=config.PROJECTION_DIM,
    proj_init_gain=config.PROJECTION_INIT_GAIN,
    use_interactions=config.USE_MULTIPLICATIVE_INTERACTIONS,
)

# Load trained weights
state_dict = torch.load("./outputs/model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Get the weights
weights_dict = model.get_linear_weights()

print("\n" + "=" * 60)
print("FEATURE WEIGHTS (Base Effects)")
print("=" * 60)
feature_weights = weights_dict["feature_weights"].numpy()
for i, feat_name in enumerate(config.FEATURE_COLS):
    print(f"{i:2d}. {feat_name:20s}: {feature_weights[i]:+.6f}")

if "interaction_weights" in weights_dict:
    print("\n" + "=" * 60)
    print("INTERACTION WEIGHTS")
    print("=" * 60)
    interaction_weights = weights_dict["interaction_weights"].numpy()
    for i, feat_name in enumerate(config.FEATURE_COLS):
        print(f"{i:2d}. {feat_name:20s}: {interaction_weights[i]:+.6f}")

print("\n" + "=" * 60)
print("CONSTRAINT VERIFICATION")
print("=" * 60)
print(f"Bias: {weights_dict['bias']:.6f}")
print()

# Check if constraints are satisfied
positive_indices = constraint_indices["positive_indices"]
negative_indices = constraint_indices["negative_indices"]

print("Positive constraint check:")
for idx in positive_indices:
    feat_name = config.FEATURE_COLS[idx]
    base_weight = feature_weights[idx]
    constraint_satisfied = base_weight > 0
    status = "✓" if constraint_satisfied else "✗"
    print(f"  {status} {feat_name:20s}: {base_weight:+.6f}")

print("\nNegative constraint check:")
for idx in negative_indices:
    feat_name = config.FEATURE_COLS[idx]
    base_weight = feature_weights[idx]
    constraint_satisfied = base_weight < 0
    status = "✓" if constraint_satisfied else "✗"
    print(f"  {status} {feat_name:20s}: {base_weight:+.6f}")

# Check sample contributions
print("\n" + "=" * 60)
print("SAMPLE CONTRIBUTION ANALYSIS")
print("=" * 60)

# Read results file
import pandas as pd

df = pd.read_csv("./outputs/results/model_results_with_contributions.csv")

# Get driver columns
driver_cols_orig = [
    col for col in df.columns if col.startswith("driver_") and col.endswith("_original")
]

# Calculate average contributions
avg_contribs = df[driver_cols_orig].mean()

print("\nAverage contributions (original scale):")
for col in driver_cols_orig:
    feat_name = col.replace("driver_", "").replace("_original", "")
    print(f"  {feat_name:20s}: {avg_contribs[col]:+10.2f}")

# Find cases where GDP has negative contribution
gdp_contrib_col = "driver_gdp_index_original"
dist_contrib_col = "driver_distribution_original"

if gdp_contrib_col in df.columns:
    negative_gdp = df[df[gdp_contrib_col] < 0]
    print(
        f"\nRows with negative GDP contribution: {len(negative_gdp)} / {len(df)} ({100*len(negative_gdp)/len(df):.1f}%)"
    )

if dist_contrib_col in df.columns:
    negative_dist = df[df[dist_contrib_col] < 0]
    print(
        f"Rows with negative distribution contribution: {len(negative_dist)} / {len(df)} ({100*len(negative_dist)/len(df):.1f}%)"
    )
