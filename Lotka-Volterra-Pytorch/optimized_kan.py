"""
Optimized Kolmogorov-Arnold Network (KAN) Implementation
This module implements an optimized KAN with various regularization and stability improvements.

Optimizations included:
1. Model complexity control
2. L2 regularization on spline coefficients
3. Smoothness penalties (second-derivative regularization)
4. Input/output normalization
5. Hybrid architecture (KAN + MLP)
6. Pruning capabilities
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np


class OptimizedKANLinear(torch.nn.Module):
    """
    Optimized KAN Linear layer with enhanced regularization and stability features.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        l2_lambda=0.0001,  # L2 regularization strength
        smoothness_lambda=0.001,  # Smoothness penalty strength
    ):
        super(OptimizedKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.l2_lambda = l2_lambda
        self.smoothness_lambda = smoothness_lambda

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """Compute the B-spline bases for the given input tensor."""
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """Compute the coefficients of the curve that interpolates the given points."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """Update grid based on data distribution."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def l2_regularization(self):
        """
        Compute L2 regularization on spline coefficients.
        Prevents overfitting by penalizing large coefficient values.
        """
        return self.l2_lambda * torch.sum(self.spline_weight ** 2)

    def smoothness_regularization(self):
        """
        Compute smoothness penalty using second-derivative approximation.
        Prevents highly oscillatory spline behavior.
        """
        # Compute second differences as approximation of second derivative
        if self.spline_weight.size(-1) < 3:
            return torch.tensor(0.0, device=self.spline_weight.device)
        
        # Second difference: coeff[i] - 2*coeff[i+1] + coeff[i+2]
        second_diff = (
            self.spline_weight[:, :, :-2] 
            - 2 * self.spline_weight[:, :, 1:-1] 
            + self.spline_weight[:, :, 2:]
        )
        return self.smoothness_lambda * torch.sum(second_diff ** 2)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Comprehensive regularization combining L1, entropy, L2, and smoothness.
        """
        # Original L1 and entropy regularization
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-10)
        regularization_loss_entropy = -torch.sum(p * p.log() + 1e-10)
        
        # Add L2 and smoothness regularization
        l2_reg = self.l2_regularization()
        smooth_reg = self.smoothness_regularization()
        
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
            + l2_reg
            + smooth_reg
        )

    @torch.no_grad()
    def prune_edges(self, threshold=0.01):
        """
        Prune edges with near-zero contribution for model simplification.
        Returns number of pruned edges.
        """
        # Compute edge importance as mean absolute weight
        edge_importance = self.spline_weight.abs().mean(dim=-1)
        mask = edge_importance > threshold
        pruned_count = (~mask).sum().item()
        
        # Zero out low-importance edges
        for i in range(self.out_features):
            for j in range(self.in_features):
                if not mask[i, j]:
                    self.spline_weight.data[i, j, :] = 0
        
        return pruned_count


class OptimizedKAN(torch.nn.Module):
    """
    Optimized Kolmogorov-Arnold Network with all enhancement features.
    
    Architecture follows heuristics:
    - Layers: 2-3
    - Width: 5-20
    - Spline grid size: 5-10
    """
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        l2_lambda=0.0001,
        smoothness_lambda=0.001,
    ):
        super(OptimizedKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                OptimizedKANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    l2_lambda=l2_lambda,
                    smoothness_lambda=smoothness_lambda,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Total regularization loss across all layers."""
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    @torch.no_grad()
    def prune_model(self, threshold=0.01):
        """
        Prune the entire model by removing low-importance edges.
        Returns total number of pruned edges.
        """
        total_pruned = sum(layer.prune_edges(threshold) for layer in self.layers)
        return total_pruned


class HybridKAN_MLP(torch.nn.Module):
    """
    Hybrid architecture: KAN for main dynamics + MLP for residuals.
    This helps learn both structured dynamics (KAN) and residual errors (MLP).
    """
    def __init__(
        self,
        kan_layers,
        mlp_hidden_dims=[16, 8],
        grid_size=5,
        spline_order=3,
        l2_lambda=0.0001,
        smoothness_lambda=0.001,
    ):
        super(HybridKAN_MLP, self).__init__()
        
        # KAN component for main dynamics
        self.kan = OptimizedKAN(
            kan_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            l2_lambda=l2_lambda,
            smoothness_lambda=smoothness_lambda,
        )
        
        # MLP component for residual learning
        mlp_layers = []
        input_dim = kan_layers[0]
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.Tanh())
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(input_dim, kan_layers[-1]))
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Weight parameter to balance KAN and MLP contributions
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Start with small MLP contribution

    def forward(self, x: torch.Tensor):
        kan_output = self.kan(x)
        mlp_output = self.mlp(x)
        # Combine outputs with learned weight (clipped for stability)
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        return (1 - alpha_clamped) * kan_output + alpha_clamped * mlp_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Total regularization from KAN component."""
        return self.kan.regularization_loss(regularize_activation, regularize_entropy)


class DataNormalizer:
    """
    Normalizes input and output data to fixed ranges.
    Essential for proper spline knot placement and stable training.
    """
    def __init__(self, data_range=(-1, 1)):
        self.data_range = data_range
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None

    def fit_input(self, data):
        """Fit normalizer to input data."""
        if isinstance(data, torch.Tensor):
            self.input_min = data.min(dim=0, keepdim=True)[0]
            self.input_max = data.max(dim=0, keepdim=True)[0]
        else:
            self.input_min = torch.tensor(np.min(data, axis=0, keepdims=True))
            self.input_max = torch.tensor(np.max(data, axis=0, keepdims=True))
        # Avoid division by zero
        self.input_max = torch.where(
            self.input_max == self.input_min, 
            self.input_min + 1, 
            self.input_max
        )

    def fit_output(self, data):
        """Fit normalizer to output data."""
        if isinstance(data, torch.Tensor):
            self.output_min = data.min(dim=0, keepdim=True)[0]
            self.output_max = data.max(dim=0, keepdim=True)[0]
        else:
            self.output_min = torch.tensor(np.min(data, axis=0, keepdims=True))
            self.output_max = torch.tensor(np.max(data, axis=0, keepdims=True))
        # Avoid division by zero
        self.output_max = torch.where(
            self.output_max == self.output_min, 
            self.output_min + 1, 
            self.output_max
        )

    def normalize_input(self, data):
        """Normalize input data to specified range."""
        if self.input_min is None:
            raise ValueError("Normalizer not fitted to input data")
        normalized = (data - self.input_min) / (self.input_max - self.input_min)
        return normalized * (self.data_range[1] - self.data_range[0]) + self.data_range[0]

    def denormalize_input(self, data):
        """Denormalize input data back to original scale."""
        if self.input_min is None:
            raise ValueError("Normalizer not fitted to input data")
        denormalized = (data - self.data_range[0]) / (self.data_range[1] - self.data_range[0])
        return denormalized * (self.input_max - self.input_min) + self.input_min

    def normalize_output(self, data):
        """Normalize output data to specified range."""
        if self.output_min is None:
            raise ValueError("Normalizer not fitted to output data")
        normalized = (data - self.output_min) / (self.output_max - self.output_min)
        return normalized * (self.data_range[1] - self.data_range[0]) + self.data_range[0]

    def denormalize_output(self, data):
        """Denormalize output data back to original scale."""
        if self.output_min is None:
            raise ValueError("Normalizer not fitted to output data")
        denormalized = (data - self.data_range[0]) / (self.data_range[1] - self.data_range[0])
        return denormalized * (self.output_max - self.output_min) + self.output_min
