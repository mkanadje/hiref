# tests/test_preprocessor.py

import pytest
import pandas as pd
import numpy as np
import os
from data.preprocessor import DataPreprocessor

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_config():
    """Return test configuration"""
    return {
        "hierarchy_cols": ["region", "state", "segment", "brand", "pack"],
        "feature_cols": ["temperature", "price", "tv_spend"],
        "target_col": "sales",
        "key_col": "sku_key",
        "date_col": "date",
    }


@pytest.fixture
def sample_df():
    """Create a small test dataframe"""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
            "region": np.random.choice(
                ["Northeast", "Southeast", "Midwest"], n_samples
            ),
            "state": np.random.choice(
                ["Northfield", "Suncoast", "Lakewood"], n_samples
            ),
            "segment": np.random.choice(["Premium", "Mid-tier", "Economy"], n_samples),
            "brand": np.random.choice(
                ["Alpine Springs", "Fresh Valley", "Value Plus"], n_samples
            ),
            "pack": np.random.choice(["8oz", "16oz", "32oz"], n_samples),
            "temperature": np.random.normal(60, 15, n_samples),
            "price": np.random.uniform(1, 10, n_samples),
            "tv_spend": np.random.uniform(0, 100, n_samples),
            "sales": np.random.uniform(100, 5000, n_samples),
        }
    )
    return df


@pytest.fixture
def preprocessor(sample_config):
    """Return initialized preprocessor"""
    return DataPreprocessor(**sample_config)


# =============================================================================
# TESTS - load_data
# =============================================================================


def test_load_data_returns_dataframe(preprocessor, sample_df, tmp_path):
    """Test that load_data returns dataframe with datetime column"""
    # Save sample df to temp file
    file_path = tmp_path / "test_data.csv"
    sample_df.to_csv(file_path, index=False)

    # Load it back
    loaded_df = preprocessor.load_data(file_path)

    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == len(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(loaded_df["date"])


# =============================================================================
# TESTS - create_key_column
# =============================================================================


def test_create_key_column_format(preprocessor, sample_df):
    """Test that key column is created with correct format"""
    df = preprocessor.create_key_column(sample_df.copy())

    assert "sku_key" in df.columns

    # Check format: should be region_state_segment_brand_pack
    first_key = df["sku_key"].iloc[0]
    parts = first_key.split("_")
    assert len(parts) == 5

    # Verify it matches the actual values
    expected_key = "_".join(
        [
            str(sample_df["region"].iloc[0]),
            str(sample_df["state"].iloc[0]),
            str(sample_df["segment"].iloc[0]),
            str(sample_df["brand"].iloc[0]),
            str(sample_df["pack"].iloc[0]),
        ]
    )
    assert first_key == expected_key


def test_create_key_column_all_rows(preprocessor, sample_df):
    """Test that key column is created for all rows"""
    df = preprocessor.create_key_column(sample_df.copy())

    assert df["sku_key"].notna().all()
    assert len(df["sku_key"]) == len(sample_df)


# =============================================================================
# TESTS - fit
# =============================================================================


def test_fit_creates_encoders(preprocessor, sample_df):
    """Test that fit populates label_encoders and vocab_sizes"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)

    # Check label encoders created for each hierarchy
    assert len(preprocessor.label_encoders) == 5
    for col in ["region", "state", "segment", "brand", "pack"]:
        assert col in preprocessor.label_encoders
        assert col in preprocessor.vocab_sizes
        assert preprocessor.vocab_sizes[col] > 0


def test_fit_creates_scalers(preprocessor, sample_df):
    """Test that fit creates feature and target scalers"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)

    assert preprocessor.feature_scaler is not None
    assert preprocessor.target_scaler is not None


def test_fit_vocab_sizes_correct(preprocessor, sample_df):
    """Test that vocab sizes match unique values in data"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)

    for col in preprocessor.hierarchy_cols:
        expected_size = df[col].nunique()
        assert preprocessor.vocab_sizes[col] == expected_size


def test_fit_returns_self(preprocessor, sample_df):
    """Test that fit returns self for method chaining"""
    df = preprocessor.create_key_column(sample_df.copy())
    result = preprocessor.fit(df)

    assert result is preprocessor


# =============================================================================
# TESTS - transform
# =============================================================================


def test_transform_output_shapes(preprocessor, sample_df):
    """Test that transform returns correct shapes"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)
    hierarchy_ids, features, targets, keys = preprocessor.transform(df)

    n_samples = len(df)
    n_features = len(preprocessor.feature_cols)

    # Check hierarchy_ids shapes
    for col in preprocessor.hierarchy_cols:
        assert hierarchy_ids[col].shape == (n_samples,)

    # Check features shape
    assert features.shape == (n_samples, n_features)

    # Check targets shape
    assert targets.shape == (n_samples,)

    # Check keys shape
    assert keys.shape == (n_samples,)


def test_transform_output_types(preprocessor, sample_df):
    """Test that transform returns numpy arrays"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)
    hierarchy_ids, features, targets, keys = preprocessor.transform(df)

    # Check hierarchy_ids are numpy arrays
    for col in preprocessor.hierarchy_cols:
        assert isinstance(hierarchy_ids[col], np.ndarray)

    # Check other outputs
    assert isinstance(features, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert isinstance(keys, np.ndarray)


def test_transform_encoded_ids_valid(preprocessor, sample_df):
    """Test that encoded IDs are valid integers within vocab size"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)
    hierarchy_ids, _, _, _ = preprocessor.transform(df)

    for col in preprocessor.hierarchy_cols:
        ids = hierarchy_ids[col]
        assert ids.dtype in [np.int32, np.int64]
        assert ids.min() >= 0
        assert ids.max() < preprocessor.vocab_sizes[col]


def test_transform_features_scaled(preprocessor, sample_df):
    """Test that features are standardized (approximately zero mean, unit variance)"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)
    _, features, _, _ = preprocessor.transform(df)

    # Check approximate standardization
    assert np.abs(features.mean()) < 0.1
    assert np.abs(features.std() - 1.0) < 0.1


# =============================================================================
# TESTS - fit_transform
# =============================================================================


def test_fit_transform_equals_fit_then_transform(
    preprocessor, sample_df, sample_config
):
    """Test that fit_transform gives same result as fit then transform"""
    df = preprocessor.create_key_column(sample_df.copy())

    # Method 1: fit_transform
    h1, f1, t1, k1 = preprocessor.fit_transform(df)

    # Method 2: fit then transform (new preprocessor)
    preprocessor2 = DataPreprocessor(**sample_config)
    df2 = preprocessor2.create_key_column(sample_df.copy())
    preprocessor2.fit(df2)
    h2, f2, t2, k2 = preprocessor2.transform(df2)

    # Compare results
    for col in preprocessor.hierarchy_cols:
        np.testing.assert_array_equal(h1[col], h2[col])
    np.testing.assert_array_almost_equal(f1, f2)
    np.testing.assert_array_almost_equal(t1, t2)
    np.testing.assert_array_equal(k1, k2)


# =============================================================================
# TESTS - split_by_date
# =============================================================================


def test_split_by_date_no_overlap(preprocessor, sample_df):
    """Test that train/val/test have no date overlap"""
    train_end = "2023-02-15"
    val_end = "2023-03-15"

    train_df, val_df, test_df = preprocessor.split_by_date(
        sample_df, train_end, val_end
    )

    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df["date"].max() <= pd.to_datetime(train_end)
        assert val_df["date"].min() > pd.to_datetime(train_end)

    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df["date"].max() <= pd.to_datetime(val_end)
        assert test_df["date"].min() > pd.to_datetime(val_end)


def test_split_by_date_chronological(preprocessor, sample_df):
    """Test that splits are in chronological order"""
    train_end = "2023-02-15"
    val_end = "2023-03-15"

    train_df, val_df, test_df = preprocessor.split_by_date(
        sample_df, train_end, val_end
    )

    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df["date"].max() < val_df["date"].min()

    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df["date"].max() < test_df["date"].min()


def test_split_by_date_complete(preprocessor, sample_df):
    """Test that all data is accounted for in splits"""
    train_end = "2023-02-15"
    val_end = "2023-03-15"

    train_df, val_df, test_df = preprocessor.split_by_date(
        sample_df, train_end, val_end
    )

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == len(sample_df)


def test_split_by_date_reset_index(preprocessor, sample_df):
    """Test that splits have reset indices"""
    train_end = "2023-02-15"
    val_end = "2023-03-15"

    train_df, val_df, test_df = preprocessor.split_by_date(
        sample_df, train_end, val_end
    )

    for df in [train_df, val_df, test_df]:
        if len(df) > 0:
            assert df.index[0] == 0
            assert df.index[-1] == len(df) - 1


# =============================================================================
# TESTS - save and load
# =============================================================================


def test_save_and_load(preprocessor, sample_df, tmp_path):
    """Test that save and load preserves preprocessor state"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)

    # Save
    save_path = tmp_path / "preprocessor.pkl"
    preprocessor.save(save_path)

    assert os.path.exists(save_path)


def test_load_restores_state(preprocessor, sample_df, tmp_path, sample_config):
    """Test that loaded preprocessor can transform data correctly"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)

    # Transform with original
    h1, f1, t1, k1 = preprocessor.transform(df)

    # Save and load into new preprocessor
    save_path = tmp_path / "preprocessor.pkl"
    preprocessor.save(save_path)

    new_preprocessor = DataPreprocessor(**sample_config)
    new_preprocessor.load(save_path)

    # Transform with loaded
    df2 = new_preprocessor.create_key_column(sample_df.copy())
    h2, f2, t2, k2 = new_preprocessor.transform(df2)

    # Compare results
    for col in preprocessor.hierarchy_cols:
        np.testing.assert_array_equal(h1[col], h2[col])
    np.testing.assert_array_almost_equal(f1, f2)
    np.testing.assert_array_almost_equal(t1, t2)


def test_load_returns_self(preprocessor, sample_df, tmp_path):
    """Test that load returns self for method chaining"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)

    save_path = tmp_path / "preprocessor.pkl"
    preprocessor.save(save_path)

    result = preprocessor.load(save_path)
    assert result is preprocessor


# =============================================================================
# TESTS - get_vocab_sizes
# =============================================================================


def test_get_vocab_sizes(preprocessor, sample_df):
    """Test that get_vocab_sizes returns correct dictionary"""
    df = preprocessor.create_key_column(sample_df.copy())
    preprocessor.fit(df)

    vocab_sizes = preprocessor.get_vocab_sizes()

    assert isinstance(vocab_sizes, dict)
    assert len(vocab_sizes) == len(preprocessor.hierarchy_cols)

    for col in preprocessor.hierarchy_cols:
        assert col in vocab_sizes
        assert vocab_sizes[col] == df[col].nunique()


# =============================================================================
# TESTS - prepare_data (integration test)
# =============================================================================


def test_prepare_data_integration(preprocessor, sample_df, tmp_path):
    """Integration test for prepare_data"""
    # Save sample data
    file_path = tmp_path / "test_data.csv"
    sample_df.to_csv(file_path, index=False)

    train_end = "2023-02-15"
    val_end = "2023-03-15"

    train_data, val_data, test_data = preprocessor.prepare_data(
        file_path, train_end, val_end
    )

    # Check that each split returns correct tuple structure
    for data in [train_data, val_data, test_data]:
        hierarchy_ids, features, targets, keys = data

        assert isinstance(hierarchy_ids, dict)
        assert isinstance(features, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert isinstance(keys, np.ndarray)

    # Check that preprocessor is fitted
    assert len(preprocessor.label_encoders) == 5
    assert preprocessor.feature_scaler is not None
    assert preprocessor.target_scaler is not None
