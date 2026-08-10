"""KKTFormer-v0 without optimizer-to-representation feedback.

The stage-4 baseline deliberately uses raw price-derived market windows rather
than signature caches.  It produces one expected-return vector per portfolio
decision with shape ``(B, N)``.  Portfolio construction is handled outside the
model by ``DifferentiablePortfolioOptimizer``.
"""

from typing import Tuple

import torch
import torch.nn as nn


class Model(nn.Module):
    """Temporal-then-asset Transformer for one portfolio decision."""

    def __init__(self, configs):
        super().__init__()
        self.num_assets = int(configs.data_pool)
        self.lookback_window = int(configs.window_size)
        self.input_dim = int(getattr(configs, "input_dim", 1))
        self.d_model = int(configs.d_model)
        self.n_heads = int(configs.n_heads)
        self.num_layers = int(configs.num_layers)
        self.ff_dim = int(configs.ff_dim)
        self.dropout = float(configs.dropout)

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.time_position = nn.Parameter(
            torch.zeros(1, self.lookback_window, self.d_model)
        )
        self.asset_embedding = nn.Embedding(self.num_assets, self.d_model)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=self.num_layers
        )

        asset_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.asset_encoder = nn.TransformerEncoder(
            asset_layer, num_layers=self.num_layers
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.return_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        nn.init.normal_(self.time_position, mean=0.0, std=0.02)

    def forward(self, market_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a market window and return ``(hidden, mu_hat)``.

        Args:
            market_window: ``(B, W, N, F)``.  A three-dimensional
                ``(B, W, N)`` tensor is accepted and treated as ``F=1``.

        Returns:
            hidden: Asset representations with shape ``(B, N, d_model)``.
            mu_hat: One expected-return vector with shape ``(B, N)``.
        """

        if not isinstance(market_window, torch.Tensor):
            raise TypeError("market_window must be a torch.Tensor")
        if market_window.ndim == 3:
            market_window = market_window.unsqueeze(-1)
        if market_window.ndim != 4:
            raise ValueError("market_window must have shape (B, W, N, F)")

        batch_size, window, num_assets, feature_dim = market_window.shape
        if window != self.lookback_window:
            raise ValueError(
                f"expected lookback window {self.lookback_window}, got {window}"
            )
        if num_assets != self.num_assets:
            raise ValueError(
                f"expected {self.num_assets} assets, got {num_assets}"
            )
        if feature_dim != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {feature_dim}"
            )
        if not torch.isfinite(market_window).all():
            raise ValueError("market_window contains NaN or infinite values")

        # Apply temporal attention independently to each asset.
        x = self.input_projection(market_window)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * num_assets, window, self.d_model)
        x = x + self.time_position[:, :window]
        x = self.temporal_encoder(x)
        x = x.mean(dim=1)
        x = x.reshape(batch_size, num_assets, self.d_model)

        # Then model cross-asset interactions with ordinary self-attention.
        asset_ids = torch.arange(num_assets, device=market_window.device)
        x = x + self.asset_embedding(asset_ids).unsqueeze(0)
        hidden = self.final_norm(self.asset_encoder(x))
        mu_hat = self.return_head(hidden).squeeze(-1)
        return hidden, mu_hat
