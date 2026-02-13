import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedLinear(nn.Module):
    """
    Linear layer with monotonicity constraints on weights

    Uses parameterization approach:
    - Learn unconstrained parameter theta
    - Transforms theta -> W (constrained weights) in the forward pass
    - W is always recomputed, ensuring constraints are satisfied

    Transformation functions:
    - Positive constraint: w = softplus(theta) (always > 0)
    - Negative constraint: w  = -softplus(theta) (always < 0)
    - Unconstrained: w = theta (no transformation)
    """

    def __init__(
        self,
        in_features,
        out_features,
        constraint_indices=None,
        bias=True,
        initial_bias=0.0,
    ):
        """
        Args:
            in_features: int - Number of input features
            out_features: int - Number of output features (usually 1)
            constraint_indices: dict or None
                If dict, must contain:
                    - 'positive_indices': List[int] - Indices that must be positive
                    - 'negative_indices': List[int] - Indices that must be negative
                    - 'unconstrained_indices': List[int] - Indices with no constraint
                if None, all weight are unconstrained (stnadard linear layer)
            bias: bool - Whether to include bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.constraint_indices = constraint_indices
        # Define learnable paramaters
        # Shape: (out_features, in_features) -  same as nn.Linear
        self.theta = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            with torch.no_grad():
                self.bias.fill_(initial_bias)
        else:
            self.register_parameter("bias", None)

    def get_constrained_weights(self):
        """
        Transform unconstrained theta -> constrained W

        Returns:
            torch.Tensor of shape (out_features, in_features) with constraints applied
        """
        if self.constraint_indices is None:
            return self.theta
        # Create a copy of theta
        W = self.theta.clone()

        # Apply positive constraints: w = softplus(theta)
        pos_idx = self.constraint_indices["positive_indices"]
        if len(pos_idx) > 0:
            W[:, pos_idx] = F.softplus(self.theta[:, pos_idx])

        # Apply negative constraints: w = -softplus(theta)
        neg_idx = self.constraint_indices["negative_indices"]
        if len(neg_idx) > 0:
            W[:, neg_idx] = -F.softplus(self.theta[:, neg_idx])

        # No change is requried for unconstrained indices
        return W

    def forward(self, x):
        """
        Forward pass with constrained weights
        Args:
            x: torch.Tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor of shape (batch_size, out_features)
        """
        # Compute unconstrained weights
        W = self.get_constrained_weights()
        # Standard linear operation
        return F.linear(x, W, self.bias)

    def extra_repr(self):
        """String representation for print(models)"""
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        if self.bias is None:
            s += ", bias=False"
        if self.constraint_indices is not None:
            n_pos = len(self.constraint_indices["positive_indices"])
            n_neg = len(self.constraint_indices["negative_indices"])
            n_unc = len(self.constraint_indices["unconstrained_indices"])
            s += f", constraints=(+{n_pos}, -{n_neg}, +-{n_unc})"
        else:
            s += ", constraints=None"
        return s
