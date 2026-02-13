import torch
import torch.nn as nn

from hier_reg.models.constrained_linear import ConstrainedLinear


class HierarchicalModel(nn.Module):
    def __init__(
        self,
        vocab_sizes,
        embedding_dims,
        n_features,
        use_interactions=False,
        projection_dim=None,
        proj_init_gain=None,
        constraint_indices=None,
        initial_bias=0.0,
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        self.n_features = n_features
        self.proj_init_gain = proj_init_gain
        self.use_interactions = use_interactions

        # Define embedding layers with specified input and output shapes
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(vocab_sizes[col], embedding_dims[col])
                for col in vocab_sizes.keys()
            }
        )
        total_embedding_dim = sum(self.embedding_dims.values())
        if use_interactions:
            assert projection_dim is not None
            self.projection_dim = projection_dim
            self.emb_projection = nn.Linear(total_embedding_dim, self.projection_dim)
            nn.init.xavier_uniform_(
                self.emb_projection.weight, gain=self.proj_init_gain
            )
            nn.init.zeros_(self.emb_projection.bias)
        else:
            self.projection_dim = 0
            # self.linear = nn.Linear(total_embedding_dim + n_features, 1)
        interaction_dim = self.projection_dim if use_interactions else 0
        combined_dim = total_embedding_dim + self.n_features + interaction_dim
        if constraint_indices is not None:
            # constraint_indices apply to FEATURES only (last n_features dimensions)
            # We need to offset indices by total_embedding_dim
            adjusted_constraints = self._adjust_constraint_indices(
                constraint_indices, total_embedding_dim
            )
            self.linear = ConstrainedLinear(
                combined_dim,
                1,
                constraint_indices=adjusted_constraints,
                initial_bias=initial_bias,
            )
            print(f"Using ConstrainedLinear layer with business logic constraints")
        else:
            self.linear = nn.Linear(total_embedding_dim + n_features, 1)
            if initial_bias != 0.0:
                with torch.no_grad():
                    self.linear.bias.fill_(initial_bias)

            print("Using standard Linear layer (unconstrained weights)")

    def _adjust_constraint_indices(self, constraint_indices, embedding_dim):
        """
        Adjust constraint indices to account for embedding dimensions.

        Constraint indices from confi refer to FEATURE positions.
        but in linear layer, features come AFTER embeddings
        Example:
            - Embedding: dimensions 0-34 (35 total)
            - Features: dimensions 35-46 (12 total)
            - Feature index 0 (temperature) -> linear layer index 35

        Args:
            constraint_indices: dict with "positive_indices", "negative_indices", "unconstrained_indices"

        Returns:
            dict with adjusted indices
        """
        # We will not constrain interaction effects
        return {
            "positive_indices": [
                idx + embedding_dim for idx in constraint_indices["positive_indices"]
            ],
            "negative_indices": [
                idx + embedding_dim for idx in constraint_indices["negative_indices"]
            ],
            "unconstrained_indices": [
                idx + embedding_dim
                for idx in constraint_indices["unconstrained_indices"]
            ],
        }

    def forward(self, hierarchy_ids, features):
        """
        Args:
            hierarchy_ids: Dict[str, torch.Tensor]
                Dictionary mapping hierarchy column names to ID tensors
                Example: {"region": tensor([0, 2, 5]), "state": tensor([0, 3, 1])...}
                Each tensor has a shape of (batch_size,) with dtype=torch.long
            features: torch.Tensor
                Numerical feature values tensor of shape (batch_size, n_features)
                Example: (1024, 12) for batch_size of 1024 and feature values 12
        Returns:
            torch.Tensor
                Predictions of shape (batch_size,)
                Example: tensor([125.3, 84.4, 23.1,....]) for batch_size
        """
        embedding_list = []
        for col in self.vocab_sizes.keys():
            emb_layer = self.embeddings[col]
            emb = emb_layer(hierarchy_ids[col])
            embedding_list.append(emb)
        # Concatenate all embeddings along the feature dims
        # Input: [(B, 4), (B, 8), (B, 3), (B, 16), (B, 4)]
        # Ouptut: (B, 35)
        all_embeddings = torch.cat(embedding_list, dim=1)

        if self.use_interactions:
            em_projs = self.emb_projection(
                all_embeddings
            )  # This should return projections of the shape (input_dims)
            interactions = em_projs * features
            combined = torch.cat([all_embeddings, features, interactions], dim=1)
        else:
            # Concat embedding with numerical features
            combined = torch.cat([all_embeddings, features], dim=1)
        # Pass through the linear layer
        output = self.linear(combined)
        output = output.squeeze(-1)
        return output

    def get_linear_weights(self):
        """
        Extract the liner weights for interpretation

        Returns:
            dict: Dictionary containing
                - 'weights': tensor of shape (47,) in this example. 35 embedding and 12 features
                - 'bias': scaler tensor - the bias term
                - 'embedding_weights': tensor of shape (35,)
                - 'feature_weights': tensor of shape (12,)
        """
        if isinstance(self.linear, ConstrainedLinear):
            weights = self.linear.get_constrained_weights().data.squeeze()
            bias = self.linear.bias.data.item()
        else:
            weights = self.linear.weight.data.squeeze()  # (47,)
            bias = self.linear.bias.data.item()  # scaler

        total_embedding_dim = sum(self.embedding_dims.values())  # 35
        interaction_start = total_embedding_dim + self.n_features
        embedding_weights = weights[:total_embedding_dim]  # (35, )
        feature_weights = weights[total_embedding_dim:interaction_start]  # (12, )
        results = {
            "weights": weights,
            "bias": bias,
            "embedding_weights": embedding_weights,
            "feature_weights": feature_weights,
        }
        if self.use_interactions:
            results["interaction_weights"] = weights[interaction_start:]
        return results

    def get_projection_weights(self):
        if not self.use_interactions:
            return None
        weight = self.emb_projection.weight.detach()
        bias = self.emb_projection.bias.detach()
        return {"weight": weight, "bias": bias}

    def get_prediction_contributions(self, hierarchy_ids, features):
        """
        Break down predictions into contributions from each component
        Args:
            hierarchy_ids: Dict[str, torch.Tensor] - same as forward()
            features: torch.Tensor - same as forward()
        Returns:
            dict: Dictionary containing:
                - 'total_prediction': Final prediction value
                - 'embedding_econtribution': Contribution from all embeddings
                - 'feature_contribution': Contribution from numerical features
                - 'bias_contribution': Contribution from bias term
                - 'feature_breakdown': Dict mapping feature index to contribution
        """
        embedding_list = []
        for col in self.vocab_sizes.keys():
            emb = self.embeddings[col](hierarchy_ids[col])
            embedding_list.append(emb)

        all_embeddings = torch.cat(embedding_list, dim=1)  # (B, 35)
        combined = torch.cat([all_embeddings, features], dim=1)  # (B, 47)
        if self.use_interactions:
            emb_projected = self.emb_projection(all_embeddings)
            interactions = emb_projected * features
            combined = torch.cat([combined, interactions], dim=1)

        if isinstance(self.linear, ConstrainedLinear):
            weights = self.linear.get_constrained_weights().data.squeeze()
        else:
            weights = self.linear.weight.data.squeeze()  # (47, )
        bias = self.linear.bias.data  # scaler

        # calculate contributions
        # contribution = weight * input_value
        total_embedding_dim = sum(self.embedding_dims.values())
        interaction_start = total_embedding_dim + self.n_features
        # Split toal emebdding
        emb_part = combined[:, :total_embedding_dim]  # (B, 35)
        feat_part = combined[:, total_embedding_dim:interaction_start]  # (B, 12)

        # Split weights
        emb_weights = weights[:total_embedding_dim]
        feat_weights = weights[total_embedding_dim:interaction_start]

        # Elementwise multiply and sum
        embedding_contribution = (emb_part * emb_weights).sum(dim=1)
        feat_contribution = (feat_part * feat_weights).sum(dim=1)

        if self.use_interactions:
            interaction_part = combined[:, interaction_start:]
            interaction_weight = weights[interaction_start:]
            interaction_contribution = (interaction_part * interaction_weight).sum(
                dim=1
            )

        # Feature breakdown (per feature)
        feature_breakdown = {}
        interaction_breakdown = {}
        for i in range(self.n_features):
            feature_breakdown[f"feature_{i}"] = (
                feat_part[:, i] * feat_weights[i]
            ).detach()
            if self.use_interactions:
                interaction_breakdown[f"interaction_{i}"] = (
                    interaction_part[:, i] * interaction_weight[i]
                ).detach()

        if self.use_interactions:
            total_prediction = (
                embedding_contribution
                + feat_contribution
                + interaction_contribution
                + bias
            )
        else:
            total_prediction = embedding_contribution + feat_contribution + bias

        return {
            "total_prediction": total_prediction.detach(),
            "embedding_contribution": embedding_contribution.detach(),
            "feature_contribution": feat_contribution.detach(),
            "bias_contribution": bias.item(),
            "feature_breakdown": feature_breakdown,
            "interaction_breakdown": interaction_breakdown,
        }

    def get_embedding_weights(self):
        embedding_weights = {}
        for col, emb_layer in self.embeddings.items():
            embedding_weights[col] = emb_layer.weight.data.clone()
        return embedding_weights
