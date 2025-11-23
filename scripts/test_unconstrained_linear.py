# In a test script
import sys

sys.path.append(".")

import config
from models.hierarchical_model import HierarchicalModel

# Mock constraint indices (normally from preprocessor)
constraint_indices = {
    "positive_indices": [0, 2, 4, 6, 7, 8, 9, 10],
    "negative_indices": [1, 3, 5],
    "unconstrained_indices": [],
}

# Test with constraints
vocab_sizes = {"region": 5, "state": 25, "segment": 3, "brand": 36, "pack": 9}
model = HierarchicalModel(
    vocab_sizes=vocab_sizes,
    embedding_dims=config.EMBEDDING_DIMS,
    n_features=len(config.FEATURE_COLS),
    constraint_indices=constraint_indices,
)

print("\nModel structure:")
print(model)

# Check that linear layer is ConstrainedLinear
print(f"\nLinear layer type: {type(model.linear)}")
print(f"Linear layer: {model.linear}")
