"""KKTFormer-v0 and optimizer-informed decision feedback models.

The stage-4 baseline deliberately uses raw price-derived market windows rather
than signature caches.  It produces one expected-return vector per portfolio
decision with shape ``(B, N)``.  Portfolio construction is handled outside the
model by ``DifferentiablePortfolioOptimizer``.
"""

from typing import Dict, Optional, Tuple

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

    def encode_market_window(self, market_window: torch.Tensor) -> torch.Tensor:
        """Encode a market window into asset representations.

        Args:
            market_window: ``(B, W, N, F)``.  A three-dimensional
                ``(B, W, N)`` tensor is accepted and treated as ``F=1``.
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
        return hidden

    def predict_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project asset representations to one expected-return vector."""

        if hidden.ndim != 3 or hidden.shape[-2:] != (
            self.num_assets,
            self.d_model,
        ):
            raise ValueError(
                "hidden must have shape (B, num_assets, d_model)"
            )
        mu_hat = self.return_head(hidden).squeeze(-1)
        return mu_hat

    def forward(self, market_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a market window and return ``(hidden, mu_hat)``."""

        hidden = self.encode_market_window(market_window)
        mu_hat = self.predict_from_hidden(hidden)
        return hidden, mu_hat


class DecisionAwareAssetAttention(nn.Module):
    """Asset attention with a KKT-derived additive attention bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.context_projection = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.context_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.bias_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        hidden: torch.Tensor,
        decision_context: torch.Tensor,
        decision_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_assets, _ = hidden.shape
        context = self.context_projection(decision_context)
        context = context * self.context_gate(decision_context)
        tokens = hidden + context

        q = self.query(tokens).view(
            batch_size, num_assets, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(tokens).view(
            batch_size, num_assets, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(tokens).view(
            batch_size, num_assets, self.n_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if decision_bias is not None:
            if decision_bias.ndim == 3:
                decision_bias = decision_bias.unsqueeze(1)
            if decision_bias.shape[-2:] != (num_assets, num_assets):
                raise ValueError("decision_bias must have shape (..., N, N)")
            scores = scores + self.bias_scale * decision_bias
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, num_assets, self.d_model
        )
        output = self.dropout(self.output(output))
        return self.norm(hidden + output)


class DecisionAwareModel(nn.Module):
    """Two-step KKTFormer with dual or Jacobian decision feedback."""

    def __init__(self, configs, feedback_mode: str = "dual"):
        super().__init__()
        feedback_mode = str(feedback_mode).lower()
        if feedback_mode not in {"dual", "jacobian"}:
            raise ValueError("feedback_mode must be dual or jacobian")
        self.feedback_mode = feedback_mode
        self.backbone = Model(configs)
        self.num_assets = self.backbone.num_assets
        self.d_model = self.backbone.d_model
        self.factor_dim = int(getattr(configs, "factor_dim", 3))
        self.attention = DecisionAwareAssetAttention(
            d_model=self.d_model,
            n_heads=self.backbone.n_heads,
            dropout=self.backbone.dropout,
        )
        self.refined_norm = nn.LayerNorm(self.d_model)
        self.decision_context_encoder = nn.Sequential(
            nn.Linear(4 + self.factor_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )

    def initial_forward(
        self, market_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone.encode_market_window(market_window)
        return hidden, self.backbone.predict_from_hidden(hidden)

    @staticmethod
    def _normalise_bias(bias: torch.Tensor) -> torch.Tensor:
        scale = bias.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return (bias / scale).clamp(-5.0, 5.0)

    def refine(
        self,
        hidden: torch.Tensor,
        kkt_state: Dict[str, torch.Tensor],
        factor_exposure: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = kkt_state["weights"]
        pressure = kkt_state["pressure"]
        active_lower = kkt_state["active_lower"].to(hidden.dtype)
        active_upper = kkt_state["active_upper"].to(hidden.dtype)
        if factor_exposure is None:
            factor_exposure = torch.zeros(
                hidden.shape[0],
                self.num_assets,
                self.factor_dim,
                dtype=hidden.dtype,
                device=hidden.device,
            )
        else:
            if factor_exposure.shape[-1] != self.factor_dim:
                raise ValueError(
                    f"factor_exposure must have last dimension {self.factor_dim}"
                )
            factor_exposure = factor_exposure.to(dtype=hidden.dtype)

        context_features = torch.cat(
            [
                weights.unsqueeze(-1),
                pressure.unsqueeze(-1),
                active_lower.unsqueeze(-1),
                active_upper.unsqueeze(-1),
                factor_exposure,
            ],
            dim=-1,
        )
        decision_context = self.decision_context_encoder(context_features)

        if self.feedback_mode == "dual":
            pressure_norm = self._normalise_bias(pressure)
            decision_bias = pressure_norm.unsqueeze(-1) * pressure_norm.unsqueeze(-2)
        else:
            decision_bias = self._normalise_bias(kkt_state["jacobian"])

        refined_hidden = self.attention(
            hidden,
            decision_context,
            decision_bias=decision_bias,
        )
        refined_hidden = self.refined_norm(refined_hidden)
        return refined_hidden, self.backbone.predict_from_hidden(refined_hidden)

    def forward(
        self,
        market_window: torch.Tensor,
        kkt_state: Dict[str, torch.Tensor],
        factor_exposure: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden0, mu0 = self.initial_forward(market_window)
        hidden1, mu1 = self.refine(hidden0, kkt_state, factor_exposure)
        return hidden0, mu0, hidden1, mu1
