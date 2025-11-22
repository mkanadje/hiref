# =============================================================================
# DATA PATHS
# =============================================================================
DATA_PATH = "./sample_data/fmcg_hierarchical_data.csv"
MODEL_SAVE_PATH = "./outputs/model.pt"
PREPROCESSOR_SAVE_PATH = "./outputs/preprocessor.pkl"
RESULTS_PATH = "./outputs/results/"

# =============================================================================
# HIERARCHY CONFIGURATION
# =============================================================================
HIERARCHY_COLS = ["region", "state", "segment", "brand", "pack"]

# Embedding dimensions for each hierarchy level
EMBEDDING_DIMS = {
    "region": 4,  # 5 unique regions
    "state": 8,  # 25 unique states
    "segment": 3,  # 3 unique segments
    "brand": 16,  # 36 unique brands
    "pack": 4,  # 9 unique pack sizes
}

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
FEATURE_COLS = [
    # Weather features
    "temperature",
    "precipitation",
    # Macroeconomic features
    "gdp_index",
    "unemployment_rate",
    "consumer_confidence",
    # SKU-level features
    "price",
    "distribution",
    # Promotional features
    "tv_spend",
    "digital_spend",
    "trade_spend",
    "discount_pct",
]

TARGET_COL = "sales"

# Key column - will be generated dynamically by concatenating HIERARCHY_COLS
# Example: "Northeast_Northfield_Premium_Alpine Springs_8oz"
KEY_COL = "sku_key"
DATE_COL = "date"

# =============================================================================
# TRAIN/VAL/TEST SPLIT (Chronological)
# =============================================================================
# Train: 2019-01 to 2023-06
# Val:   2023-07 to 2024-03
# Test:  2024-04 to 2024-12

TRAIN_END_DATE = "2023-06-30"
VAL_END_DATE = "2024-03-31"
# Test is everything after VAL_END_DATE

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
WEIGHT_DECAY = 1e-5  # L2 regularization

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "mps"  # 'cuda' or 'cpu' - will auto-fallback to cpu if cuda unavailable

# =============================================================================
# RANDOM SEED
# =============================================================================
RANDOM_SEED = 42
