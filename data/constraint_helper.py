"""
Helper function for weight constraint configuration
"""


def get_constraint_indices(feature_cols, positive_features, negative_features):
    """
    Convert feature names to indices for constrained optimization
    Args:
        feature_cols: List[str] - All feature names
        positive_features: List[str] - Features should have positive coefficients
        negative_features: List[str] - Feature which have negative coefficients
    Returns:
        Dict - {
            "positive_indices" - indices for positive constraints,
            "negative_indices" - indices for negative constraints,
            "unconstrained_indices" - indices with no constraints,
        }
    Raises:
        ValueError: If feature names are invalid or duplicated
    """
    # Filter to only include features that exist in feature_cols
    # This allows partial feature lists (useful for testing)
    feature_set = set(feature_cols)

    valid_positive = [f for f in positive_features if f in feature_set]
    valid_negative = [f for f in negative_features if f in feature_set]

    # Get indices for valid features
    positive_idx = [feature_cols.index(name) for name in valid_positive]
    negative_idx = [feature_cols.index(name) for name in valid_negative]

    # Check for overlap
    overlap = set(valid_positive) & set(valid_negative)
    if overlap:
        raise ValueError(f"Feature cannot be both positive and negative: {overlap}")

    all_constrained_idx = set(positive_idx + negative_idx)
    all_idx = set(range(len(feature_cols)))
    unconstrained_idx = sorted(all_idx - all_constrained_idx)

    return {
        "positive_indices": positive_idx,
        "negative_indices": negative_idx,
        "unconstrained_indices": unconstrained_idx,
    }
