import torch
import torch.nn as nn
import config
from hier_reg.models.constrained_linear import ConstrainedLinear


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        early_stopping_patience=10,
        checkpoint_path="./outputs/best_model.pt",
        preprocessor=None,  # For inverse scaling metrics
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_path = checkpoint_path
        self.preprocessor = preprocessor
        self.criterion = nn.MSELoss()

    def _move_to_device(self, hierarchy_ids, features, targets):
        for col in hierarchy_ids.keys():
            hierarchy_ids[col] = hierarchy_ids[col].to(self.device)
        features = features.to(self.device)
        targets = targets.to(self.device)
        return hierarchy_ids, features, targets

    def train_epoch(self):
        self.model.train()  # Set to training mode
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # 1. Extract data
            hierarchy_ids, features, targets, keys = batch

            # 2. Move data to device (gpu or cpu)
            hierarchy_ids, features, targets = self._move_to_device(
                hierarchy_ids, features, targets
            )

            # 3. Zero gradients
            self.optimizer.zero_grad()

            # 4. Forward pass
            predictions = self.model(hierarchy_ids, features)

            # 5. Calculate loss
            loss = self.criterion(predictions, targets)
            if hasattr(self.model, "use_interactions") and self.model.use_interactions:
                if config.INTERACTION_L2_LAMBDA > 0:
                    if isinstance(self.model.linear, ConstrainedLinear):
                        all_weights = (
                            self.model.linear.get_constrained_weights().squeeze()
                        )
                    else:
                        all_weights = self.model.linear.weights().squeeze()
                    total_embedding_dim = sum(self.model.embedding_dims.values())
                    interaction_start = total_embedding_dim + self.model.n_features
                    interaction_weights = all_weights[interaction_start:]
                    interaction_l2 = torch.sum(interaction_weights**2)
                    loss = loss + config.INTERACTION_L2_LAMBDA * interaction_l2

            # 6. Backward pass
            loss.backward()

            # 7. Update weights
            self.optimizer.step()

            # 8. Accumulate loss
            total_loss += loss.item()
            num_batches += 1
        # 9. Return average loss
        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        """
        Validate the model for each epoch. We calculate gradient based on the
        training data. However, validation performance is used to decide
        whether we should stop training or are overfitting.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in self.val_loader:
                hierarchy_ids, features, targets, keys = batch
                hierarchy_ids, features, targets = self._move_to_device(
                    hierarchy_ids, features, targets
                )
                predictions = self.model(hierarchy_ids, features)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
                all_predictions.append(predictions)
                all_targets.append(targets)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        from hier_reg.training.metrics import calculate_metrics

        metrics = calculate_metrics(
            all_predictions,
            all_targets,
            scaler=self.preprocessor.target_scaler if self.preprocessor else None,
        )

        avg_loss = total_loss / num_batches
        return avg_loss, metrics

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.checkpoint_path
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.checkpoint_path
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0  # Count epochs without improvement
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }
        for epoch in range(self.num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1} / {self.num_epochs}")
            print(f"{'=' * 50}")
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            val_loss, val_metrics = self.validate_epoch()
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val RMSE: {val_metrics['rmse']:.2f}")
            print(f"Val MAE: {val_metrics['mae']:.2f}")
            print(f"Val MAPE: {val_metrics['mape']:.2f}%")
            print(f"Val R2: {val_metrics['r2']:.2f}")

            if val_loss < best_val_loss:
                print(f"Validation improved({best_val_loss:.4f} -> {val_loss:.4f})")
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint()
                print(f"Model saved to {self.checkpoint_path}")
            else:
                patience_counter += 1
                print(
                    f"No improvement (patience: {patience_counter} / {self.early_stopping_patience})"
                )

            if patience_counter >= self.early_stopping_patience:
                print(f"\n{'=' * 50}")
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"{'=' * 50}")
                break
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
        print(f"\nLoading the best model from {self.checkpoint_path}")
        self.load_checkpoint()
        return history
