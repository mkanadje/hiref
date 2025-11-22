# tests/test_hierarchical_model.py

import pytest
import torch
import torch.nn as nn
import numpy as np
from models.hierarchical_model import HierarchicalModel
from config import EMBEDDING_DIMS, HIERARCHY_COLS, FEATURE_COLS

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def vocab_sizes():
    """Sample vocabulary sizes for testing"""
    return {
        "region": 5,
        "state": 25,
        "segment": 3,
        "brand": 36,
        "pack": 9,
    }


@pytest.fixture
def embedding_dims():
    """Sample embedding dimensions for testing"""
    return {
        "region": 4,
        "state": 8,
        "segment": 3,
        "brand": 16,
        "pack": 4,
    }


@pytest.fixture
def n_features():
    """Number of features"""
    return 12


@pytest.fixture
def model(vocab_sizes, embedding_dims, n_features):
    """Create a test model instance"""
    torch.manual_seed(42)
    return HierarchicalModel(vocab_sizes, embedding_dims, n_features)


@pytest.fixture
def sample_batch(vocab_sizes, n_features):
    """Create a sample batch of data"""
    torch.manual_seed(42)
    batch_size = 32

    hierarchy_ids = {
        "region": torch.randint(0, vocab_sizes["region"], (batch_size,)),
        "state": torch.randint(0, vocab_sizes["state"], (batch_size,)),
        "segment": torch.randint(0, vocab_sizes["segment"], (batch_size,)),
        "brand": torch.randint(0, vocab_sizes["brand"], (batch_size,)),
        "pack": torch.randint(0, vocab_sizes["pack"], (batch_size,)),
    }
    features = torch.randn(batch_size, n_features)

    return hierarchy_ids, features


@pytest.fixture
def single_sample(vocab_sizes, n_features):
    """Create a single sample of data"""
    torch.manual_seed(42)

    hierarchy_ids = {
        "region": torch.tensor([2]),
        "state": torch.tensor([15]),
        "segment": torch.tensor([1]),
        "brand": torch.tensor([8]),
        "pack": torch.tensor([3]),
    }
    features = torch.randn(1, n_features)

    return hierarchy_ids, features


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_model_initialization(vocab_sizes, embedding_dims, n_features):
    """Test that model initializes correctly"""
    model = HierarchicalModel(vocab_sizes, embedding_dims, n_features)

    assert isinstance(model, nn.Module)
    assert model.vocab_sizes == vocab_sizes
    assert model.embedding_dims == embedding_dims
    assert model.n_features == n_features


def test_embedding_layers_created(model, vocab_sizes, embedding_dims):
    """Test that all embedding layers are created with correct dimensions"""
    assert isinstance(model.embeddings, nn.ModuleDict)

    for col in vocab_sizes.keys():
        assert col in model.embeddings
        emb_layer = model.embeddings[col]
        assert isinstance(emb_layer, nn.Embedding)
        assert emb_layer.num_embeddings == vocab_sizes[col]
        assert emb_layer.embedding_dim == embedding_dims[col]


def test_linear_layer_dimensions(model, embedding_dims, n_features):
    """Test that linear layer has correct input and output dimensions"""
    total_embedding_dim = sum(embedding_dims.values())
    expected_input_dim = total_embedding_dim + n_features

    assert isinstance(model.linear, nn.Linear)
    assert model.linear.in_features == expected_input_dim
    assert model.linear.out_features == 1


def test_module_dict_contains_all_hierarchies(model, vocab_sizes):
    """Test that ModuleDict contains all hierarchy columns"""
    for col in vocab_sizes.keys():
        assert col in model.embeddings


def test_model_stores_config(model, vocab_sizes, embedding_dims, n_features):
    """Test that model correctly stores configuration"""
    assert model.vocab_sizes == vocab_sizes
    assert model.embedding_dims == embedding_dims
    assert model.n_features == n_features


# =============================================================================
# FORWARD PASS TESTS
# =============================================================================


def test_forward_single_sample(model, single_sample):
    """Test forward pass with single sample"""
    hierarchy_ids, features = single_sample
    output = model(hierarchy_ids, features)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1,)
    assert output.dtype == torch.float32


def test_forward_batch(model, sample_batch):
    """Test forward pass with batch of samples"""
    hierarchy_ids, features = sample_batch
    batch_size = features.shape[0]

    output = model(hierarchy_ids, features)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size,)
    assert output.dtype == torch.float32


def test_output_shape_is_correct(model, vocab_sizes, n_features):
    """Test output shape matches batch size"""
    for batch_size in [1, 16, 32, 128, 1024]:
        hierarchy_ids = {
            col: torch.randint(0, vocab_sizes[col], (batch_size,))
            for col in vocab_sizes.keys()
        }
        features = torch.randn(batch_size, n_features)

        output = model(hierarchy_ids, features)
        assert output.shape == (batch_size,)


def test_output_dtype(model, sample_batch):
    """Test output has correct dtype"""
    hierarchy_ids, features = sample_batch
    output = model(hierarchy_ids, features)

    assert output.dtype == torch.float32


def test_forward_different_batch_sizes(model, vocab_sizes, n_features):
    """Test forward pass works with different batch sizes"""
    batch_sizes = [1, 5, 32, 100, 1024]

    for batch_size in batch_sizes:
        hierarchy_ids = {
            col: torch.randint(0, vocab_sizes[col], (batch_size,))
            for col in vocab_sizes.keys()
        }
        features = torch.randn(batch_size, n_features)

        output = model(hierarchy_ids, features)
        assert output.shape == (batch_size,)


def test_forward_deterministic(model, sample_batch):
    """Test that same input produces same output (deterministic)"""
    hierarchy_ids, features = sample_batch

    # Run forward pass twice with same input
    model.eval()  # Set to eval mode to ensure deterministic behavior
    with torch.no_grad():
        output1 = model(hierarchy_ids, features)
        output2 = model(hierarchy_ids, features)

    # Outputs should be identical
    assert torch.allclose(output1, output2)


# =============================================================================
# WEIGHT EXTRACTION TESTS
# =============================================================================


def test_get_linear_weights_returns_dict(model):
    """Test that get_linear_weights returns correct dictionary structure"""
    weights_dict = model.get_linear_weights()

    assert isinstance(weights_dict, dict)
    assert "weights" in weights_dict
    assert "bias" in weights_dict
    assert "embedding_weights" in weights_dict
    assert "feature_weights" in weights_dict


def test_linear_weights_shapes(model, embedding_dims, n_features):
    """Test that extracted weights have correct shapes"""
    weights_dict = model.get_linear_weights()

    total_embedding_dim = sum(embedding_dims.values())
    total_dim = total_embedding_dim + n_features

    # Check shapes
    assert weights_dict["weights"].shape == (total_dim,)
    assert weights_dict["embedding_weights"].shape == (total_embedding_dim,)
    assert weights_dict["feature_weights"].shape == (n_features,)


def test_bias_is_scalar(model):
    """Test that bias is a scalar value"""
    weights_dict = model.get_linear_weights()

    assert isinstance(weights_dict["bias"], float)


def test_weights_are_tensors(model):
    """Test that weight components are tensors"""
    weights_dict = model.get_linear_weights()

    assert isinstance(weights_dict["weights"], torch.Tensor)
    assert isinstance(weights_dict["embedding_weights"], torch.Tensor)
    assert isinstance(weights_dict["feature_weights"], torch.Tensor)


# =============================================================================
# PREDICTION CONTRIBUTION TESTS
# =============================================================================


def test_get_prediction_contributions_returns_dict(model, sample_batch):
    """Test that get_prediction_contributions returns correct structure"""
    hierarchy_ids, features = sample_batch
    contrib = model.get_prediction_contributions(hierarchy_ids, features)

    assert isinstance(contrib, dict)
    assert "total_prediction" in contrib
    assert "embedding_contribution" in contrib
    assert "feature_contribution" in contrib
    assert "bias_contribution" in contrib
    assert "feature_breakdown" in contrib


def test_contribution_shapes_match_batch(model, sample_batch):
    """Test that contribution shapes match batch size"""
    hierarchy_ids, features = sample_batch
    batch_size = features.shape[0]

    contrib = model.get_prediction_contributions(hierarchy_ids, features)

    assert contrib["total_prediction"].shape == (batch_size,)
    assert contrib["embedding_contribution"].shape == (batch_size,)
    assert contrib["feature_contribution"].shape == (batch_size,)


def test_contributions_sum_correctly(model, sample_batch):
    """Test that contributions sum to total prediction (mathematically correct)"""
    hierarchy_ids, features = sample_batch
    contrib = model.get_prediction_contributions(hierarchy_ids, features)

    # total = embedding + feature + bias
    calculated_total = (
        contrib["embedding_contribution"]
        + contrib["feature_contribution"]
        + contrib["bias_contribution"]
    )

    assert torch.allclose(contrib["total_prediction"], calculated_total, atol=1e-5)


def test_feature_breakdown_count(model, sample_batch, n_features):
    """Test that feature breakdown has correct number of features"""
    hierarchy_ids, features = sample_batch
    contrib = model.get_prediction_contributions(hierarchy_ids, features)

    assert len(contrib["feature_breakdown"]) == n_features
    for i in range(n_features):
        assert f"feature_{i}" in contrib["feature_breakdown"]


def test_contributions_are_detached(model, sample_batch):
    """Test that contributions are detached (no gradients)"""
    hierarchy_ids, features = sample_batch
    contrib = model.get_prediction_contributions(hierarchy_ids, features)

    assert not contrib["total_prediction"].requires_grad
    assert not contrib["embedding_contribution"].requires_grad
    assert not contrib["feature_contribution"].requires_grad


def test_feature_breakdown_shapes(model, sample_batch):
    """Test that feature breakdown tensors have correct shapes"""
    hierarchy_ids, features = sample_batch
    batch_size = features.shape[0]

    contrib = model.get_prediction_contributions(hierarchy_ids, features)

    for feature_contrib in contrib["feature_breakdown"].values():
        assert feature_contrib.shape == (batch_size,)


# =============================================================================
# EMBEDDING WEIGHT TESTS
# =============================================================================


def test_get_embedding_weights_returns_dict(model, vocab_sizes):
    """Test that get_embedding_weights returns dict with all hierarchies"""
    emb_weights = model.get_embedding_weights()

    assert isinstance(emb_weights, dict)
    for col in vocab_sizes.keys():
        assert col in emb_weights


def test_embedding_weight_shapes(model, vocab_sizes, embedding_dims):
    """Test that embedding weights have correct shapes"""
    emb_weights = model.get_embedding_weights()

    for col in vocab_sizes.keys():
        expected_shape = (vocab_sizes[col], embedding_dims[col])
        assert emb_weights[col].shape == expected_shape


def test_embeddings_are_cloned(model):
    """Test that embeddings are cloned, not references"""
    emb_weights1 = model.get_embedding_weights()
    emb_weights2 = model.get_embedding_weights()

    # Modify one copy
    emb_weights1["region"][0, 0] = 999.0

    # Original should be unchanged
    assert not torch.allclose(emb_weights1["region"], emb_weights2["region"])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_model_with_real_config():
    """Test model initialization with actual config values"""
    from config import EMBEDDING_DIMS, FEATURE_COLS

    # Calculate vocab sizes from embedding dims (for testing)
    vocab_sizes = {
        "region": 5,
        "state": 25,
        "segment": 3,
        "brand": 36,
        "pack": 9,
    }

    n_features = len(FEATURE_COLS)

    model = HierarchicalModel(vocab_sizes, EMBEDDING_DIMS, n_features)

    # Test with a batch
    batch_size = 32
    hierarchy_ids = {
        col: torch.randint(0, vocab_sizes[col], (batch_size,))
        for col in vocab_sizes.keys()
    }
    features = torch.randn(batch_size, n_features)

    output = model(hierarchy_ids, features)

    assert output.shape == (batch_size,)


def test_gradient_flow(model, sample_batch):
    """Test that gradients flow correctly through the model"""
    hierarchy_ids, features = sample_batch
    target = torch.randn(features.shape[0])

    # Enable gradients
    model.train()

    # Forward pass
    output = model(hierarchy_ids, features)

    # Compute loss
    loss = nn.MSELoss()(output, target)

    # Backward pass
    loss.backward()

    # Check that gradients exist for linear layer
    assert model.linear.weight.grad is not None
    assert model.linear.bias.grad is not None

    # Check that gradients exist for embeddings
    for emb_layer in model.embeddings.values():
        # Note: Only accessed embeddings will have gradients
        # So we just check that grad can be computed
        assert emb_layer.weight.grad is not None


def test_model_save_load(model, tmp_path):
    """Test that model can be saved and loaded"""
    # Save model
    save_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), save_path)

    # Create new model with same architecture
    new_model = HierarchicalModel(
        model.vocab_sizes, model.embedding_dims, model.n_features
    )

    # Load weights
    new_model.load_state_dict(torch.load(save_path))

    # Compare weights
    for key in model.state_dict().keys():
        assert torch.allclose(model.state_dict()[key], new_model.state_dict()[key])


def test_train_eval_modes(model, sample_batch):
    """Test that model.train() and model.eval() work correctly"""
    hierarchy_ids, features = sample_batch

    # Test train mode
    model.train()
    assert model.training
    output_train = model(hierarchy_ids, features)

    # Test eval mode
    model.eval()
    assert not model.training
    output_eval = model(hierarchy_ids, features)

    # Outputs should be identical (no dropout in this simple model)
    assert torch.allclose(output_train, output_eval)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_minimum_vocab_sizes():
    """Test with minimum vocabulary sizes (single category per hierarchy)"""
    vocab_sizes = {"region": 1, "state": 1, "segment": 1, "brand": 1, "pack": 1}
    embedding_dims = {"region": 2, "state": 2, "segment": 2, "brand": 2, "pack": 2}
    n_features = 5

    model = HierarchicalModel(vocab_sizes, embedding_dims, n_features)

    # All IDs will be 0 (only option)
    hierarchy_ids = {
        col: torch.zeros(10, dtype=torch.long) for col in vocab_sizes.keys()
    }
    features = torch.randn(10, n_features)

    output = model(hierarchy_ids, features)
    assert output.shape == (10,)


def test_zero_features():
    """Test with zero features (embeddings only)"""
    vocab_sizes = {"region": 5, "state": 10}
    embedding_dims = {"region": 4, "state": 8}
    n_features = 0

    model = HierarchicalModel(vocab_sizes, embedding_dims, n_features)

    hierarchy_ids = {
        "region": torch.randint(0, 5, (16,)),
        "state": torch.randint(0, 10, (16,)),
    }
    features = torch.randn(16, 0)  # Empty feature tensor

    output = model(hierarchy_ids, features)
    assert output.shape == (16,)


def test_forward_vs_contributions_consistency(model, sample_batch):
    """Test that forward() and get_prediction_contributions() are consistent"""
    hierarchy_ids, features = sample_batch

    # Get output from forward pass
    model.eval()
    with torch.no_grad():
        forward_output = model(hierarchy_ids, features)

        # Get output from contributions
        contrib = model.get_prediction_contributions(hierarchy_ids, features)
        contrib_output = contrib["total_prediction"]

    # They should be identical
    assert torch.allclose(forward_output, contrib_output, atol=1e-5)


# =============================================================================
# NUMERICAL CORRECTNESS TESTS
# =============================================================================


def test_embedding_lookup_correctness(model, vocab_sizes, n_features):
    """Test that embedding lookup is working correctly"""
    # Create a batch where all samples have the same IDs
    batch_size = 10
    hierarchy_ids = {
        col: torch.full((batch_size,), fill_value=0, dtype=torch.long)
        for col in vocab_sizes.keys()
    }
    features = torch.zeros(batch_size, n_features)

    model.eval()
    with torch.no_grad():
        output = model(hierarchy_ids, features)

    # All outputs should be identical (same input)
    assert torch.allclose(output, output[0].expand(batch_size))


def test_linear_layer_computation(model, single_sample):
    """Test that linear layer computation is correct"""
    hierarchy_ids, features = single_sample

    model.eval()
    with torch.no_grad():
        # Get embeddings manually
        embedding_list = []
        for col in model.vocab_sizes.keys():
            emb = model.embeddings[col](hierarchy_ids[col])
            embedding_list.append(emb)

        all_embeddings = torch.cat(embedding_list, dim=1)
        combined = torch.cat([all_embeddings, features], dim=1)

        # Manual linear transformation
        manual_output = (
            torch.matmul(combined, model.linear.weight.t()) + model.linear.bias
        ).squeeze(-1)

        # Model output
        model_output = model(hierarchy_ids, features)

        # Should be identical
        assert torch.allclose(manual_output, model_output, atol=1e-5)


def test_contribution_calculation_correctness(model, single_sample):
    """Test that contribution calculation is mathematically correct"""
    hierarchy_ids, features = single_sample

    contrib = model.get_prediction_contributions(hierarchy_ids, features)
    weights_dict = model.get_linear_weights()

    # Manually calculate embedding contribution
    embedding_list = []
    for col in model.vocab_sizes.keys():
        emb = model.embeddings[col](hierarchy_ids[col])
        embedding_list.append(emb)

    all_embeddings = torch.cat(embedding_list, dim=1)

    manual_emb_contrib = torch.sum(
        all_embeddings * weights_dict["embedding_weights"], dim=1
    )
    manual_feat_contrib = torch.sum(features * weights_dict["feature_weights"], dim=1)

    # Compare
    assert torch.allclose(
        contrib["embedding_contribution"], manual_emb_contrib.detach(), atol=1e-5
    )
    assert torch.allclose(
        contrib["feature_contribution"], manual_feat_contrib.detach(), atol=1e-5
    )
