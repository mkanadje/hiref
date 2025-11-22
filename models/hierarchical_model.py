import torch
import torch.nn as nn


class HierarchicalModel(nn.Module):
    def __init__(self, vocab_sizes, embedding_dims, n_features):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        self.n_features = n_features

        # Define embedding layers with specified input and output shapes
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(vocab_sizes[col], embedding_dims[col])
                for col in vocab_sizes.keys()
            }
        )
        total_embedding_dim = sum(self.embedding_dims.values())
        self.linear = nn.Linear(total_embedding_dim + n_features, 1)

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
        weights = self.linear.weight.data.squeeze()  # (47,)
        bias = self.linear.bias.data.item()  # scaler

        total_embedding_dim = sum(self.embedding_dims.values())  # 35
        embedding_weights = weights[:total_embedding_dim]  # (35, )
        feature_weights = weights[total_embedding_dim:]  # (12, )
        return {
            "weights": weights,
            "bias": bias,
            "embedding_weights": embedding_weights,
            "feature_weights": feature_weights,
        }

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

        weights = self.linear.weight.data.squeeze()  # (47, )
        bias = self.linear.bias.data  # scaler

        # calculate contributions
        # contribution = weight * input_value
        total_embedding_dim = sum(self.embedding_dims.values())

        # Split toal emebdding
        emb_part = combined[:, :total_embedding_dim]  # (B, 35)
        feat_part = combined[:, total_embedding_dim:]  # (B, 12)

        # Split weights
        emb_weights = weights[:total_embedding_dim]
        feat_weights = weights[total_embedding_dim:]

        # Elementwise multiply and sum
        embedding_contribution = (emb_part * emb_weights).sum(dim=1)
        feat_contribution = (feat_part * feat_weights).sum(dim=1)

        # Feature breakdown (per feature)
        feature_breakdown = {}
        for i in range(self.n_features):
            feature_breakdown[f"feature_{i}"] = (
                feat_part[:, i] * feat_weights[i]
            ).detach()

        total_prediction = embedding_contribution + feat_contribution + bias

        return {
            "total_prediction": total_prediction.detach(),
            "embedding_contribution": embedding_contribution.detach(),
            "feature_contribution": feat_contribution.detach(),
            "bias_contribution": bias.item(),
            "feature_breakdown": feature_breakdown,
        }

    def get_embedding_weights(self):
        embedding_weights = {}
        for col, emb_layer in self.embeddings.items():
            embedding_weights[col] = emb_layer.weight.data.clone()
        return embedding_weights
