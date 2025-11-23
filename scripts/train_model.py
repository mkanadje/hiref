import sys

sys.path.append(".")

import torch
import config
from data.preprocessor import DataPreprocessor
from data.dataset import HierarchicalDataset
from torch.utils.data import DataLoader
from models.hierarchical_model import HierarchicalModel
from training.trainer import Trainer
from training.metrics import calculate_metrics
import config

torch.manual_seed(config.RANDOM_SEED)


def main():
    print("=" * 60)
    print("STEP 1: PREPARING DATA")
    print("=" * 60)
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        hierarchy_cols=config.HIERARCHY_COLS,
        feature_cols=config.FEATURE_COLS,
        target_col=config.TARGET_COL,
        key_col=config.KEY_COL,
        date_col=config.DATE_COL,
    )
    # Prepare Data (load, fit, transform, split)
    print(f"Loading and prepararing data from {config.DATA_PATH}")
    train_data, val_data, test_data = preprocessor.prepare_data(
        config.DATA_PATH, config.TRAIN_END_DATE, config.VAL_END_DATE
    )
    print("Data Shapes")
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        hierarchy_ids, features, target, keys = data
        print(f"{name}: {len(target)} samples, {features.shape[1]} features")

    # Create Datasets
    print("=" * 60)
    print("STEP 2: CREATING DATASETS")
    print("=" * 60)
    # Create PyTorch Datasets
    train_dataset = HierarchicalDataset(*train_data)
    val_dataset = HierarchicalDataset(*val_data)
    test_dataset = HierarchicalDataset(*test_data)
    print(f"Train Datasets: {len(train_dataset)} samples")
    print(f"Val Dataset: {len(val_dataset)} samples")
    print(f"Test Dataset: {len(test_dataset)} samples")

    print("=" * 60)
    print("STEP 3: CREATING DATALOADER")
    print("=" * 60)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("=" * 60)
    print("STEP 4: CREATING INITIALIZING MODEL")
    print("=" * 60)
    vocab_sizes = preprocessor.get_vocab_sizes()
    model = HierarchicalModel(
        vocab_sizes=vocab_sizes,
        embedding_dims=config.EMBEDDING_DIMS,
        n_features=len(config.FEATURE_COLS),
        constraint_indices=preprocessor.constraint_indices,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    print("=" * 60)
    print("STEP 5: SETTING UP OPTIMIZER")
    print("=" * 60)
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"Warning: {config.DEVICE} is not available, using CPU")
    print(f"Using device: {device}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    print("=" * 60)
    print("STEP 6: TRAINING MODEL")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.NUM_EPOCHS,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        checkpoint_path=config.MODEL_SAVE_PATH,
        preprocessor=preprocessor,
    )
    history = trainer.train()
    print("\nTraining Complete!")

    print("=" * 60)
    print("STEP 7: EVALUATING ON THE TEST SET")
    print("=" * 60)

    # Load the best model
    trainer.load_checkpoint()

    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            hierarchy_ids, features, targets, keys = batch
            hierarchy_ids, features, targets = trainer._move_to_device(
                hierarchy_ids, features, targets
            )
            predictions = model(hierarchy_ids, features)
            test_predictions.append(predictions.cpu())
            test_targets.append(targets.cpu())
    test_predictions = torch.cat(test_predictions, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    test_metrics = calculate_metrics(
        test_predictions, test_targets, scaler=preprocessor.target_scaler
    )
    print("\nTest Set Results:")
    print(f"MSE: {test_metrics['mse']:.2f}")
    print(f"RMSE: {test_metrics['rmse']:.2f}")
    print(f"MAE: {test_metrics['mae']:.2f}")
    print(
        f" MAPE: {test_metrics['mape']:.2f}% (coverage: {test_metrics['mape_coverage']:.1f}%)"
    )
    print(f" R2: {test_metrics['r2']:.4f}")

    print("=" * 60)
    print("STEP 8: SAVING ARTIFACTS")
    print("=" * 60)

    preprocessor.save(config.PREPROCESSOR_SAVE_PATH)
    print(f"Preprocessor saved to {config.PREPROCESSOR_SAVE_PATH}")

    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # Save test metrics
    import json
    import os

    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    test_metrics_json = {k: float(v) for k, v in test_metrics.items()}
    with open(f"{config.RESULTS_PATH}/test_metrics.json", "w") as f:
        json.dump(test_metrics_json, f, indent=2)
    print(f"Test metrics saved to {config.RESULTS_PATH}/test_metrics.json")
    print("\nAll artifacts saved successfully!")


if __name__ == "__main__":
    main()
