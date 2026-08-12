"""KKTFormer with SIT-aligned path, date, and asset token inputs."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Model(nn.Module):
    """SIT-aligned sequence-to-sequence decision backbone."""

    def __init__(self, configs):
        super().__init__()
        self.num_assets = int(configs.data_pool)
        self.lookback_window = int(configs.window_size)
        self.horizon = int(configs.horizon)
        self.time_feat_dim = int(getattr(configs, "time_feat_dim", 3))
        self.d_model = int(configs.d_model)
        self.n_heads = int(configs.n_heads)
        self.num_layers = int(configs.num_layers)
        self.ff_dim = int(configs.ff_dim)
        self.dropout = float(configs.dropout)
        self.signal_normalization = str(
            getattr(configs, "signal_normalization", "risk")
        ).lower()
        self.signal_scale = float(getattr(configs, "signal_scale", 0.05))
        self.signal_normalization_epsilon = float(
            getattr(configs, "signal_normalization_epsilon", 1e-6)
        )
        self.eta = float(getattr(configs, "eta", 1e-3))

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.signal_normalization not in {"risk", "none"}:
            raise ValueError("signal_normalization must be risk or none")
        if self.signal_scale <= 0:
            raise ValueError("signal_scale must be positive")
        if self.signal_normalization_epsilon <= 0:
            raise ValueError("signal_normalization_epsilon must be positive")

        self.path_projection = nn.Linear(self.lookback_window, self.d_model)
        self.date_projection = nn.Linear(self.time_feat_dim, self.d_model)
        self.asset_embedding = nn.Embedding(self.num_assets, self.d_model)
        self.concat_projection = nn.Linear(3 * self.d_model, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)

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

    def encode_inputs(
        self, log_return_path: torch.Tensor, date_feats: torch.Tensor
    ) -> torch.Tensor:
        """Encode ``(B,H,N,W)`` paths with date and asset identity features."""

        if log_return_path.ndim != 4:
            raise ValueError("log_return_path must have shape (B, H, N, W)")
        batch_size, horizon, num_assets, window = log_return_path.shape
        if horizon != self.horizon:
            raise ValueError(f"expected horizon {self.horizon}, got {horizon}")
        if window != self.lookback_window:
            raise ValueError(f"expected path width {self.lookback_window}, got {window}")
        if num_assets != self.num_assets:
            raise ValueError(f"expected {self.num_assets} assets, got {num_assets}")
        if date_feats.shape != (batch_size, horizon, self.time_feat_dim):
            raise ValueError(
                f"date_feats must have shape (B, H, {self.time_feat_dim})"
            )
        if not torch.isfinite(log_return_path).all() or not torch.isfinite(date_feats).all():
            raise ValueError("model inputs contain NaN or infinite values")

        path_emb = self.path_projection(log_return_path)
        date_emb = self.date_projection(date_feats).unsqueeze(2).expand(-1, -1, num_assets, -1)
        asset_ids = torch.arange(num_assets, device=log_return_path.device)
        asset_emb = self.asset_embedding(asset_ids).view(1, 1, num_assets, -1).expand(batch_size, horizon, -1, -1)
        x = self.input_norm(self.concat_projection(torch.cat((path_emb, date_emb, asset_emb), dim=-1)))

        # Temporal attention is causal and is applied independently per asset.
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_assets, horizon, self.d_model)
        causal_mask = torch.triu(
            torch.ones(horizon, horizon, device=x.device, dtype=torch.bool), diagonal=1
        )
        x = self.temporal_encoder(x, mask=causal_mask)
        x = x.reshape(batch_size, num_assets, horizon, self.d_model).permute(0, 2, 1, 3)

        # Asset attention is applied independently at every horizon position.
        x = x.reshape(batch_size * horizon, num_assets, self.d_model)
        x = self.final_norm(self.asset_encoder(x))
        return x.reshape(batch_size, horizon, num_assets, self.d_model)

    def predict_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project ``(..., N, d_model)`` representations to raw signals."""

        if hidden.ndim < 3 or hidden.shape[-2:] != (
            self.num_assets,
            self.d_model,
        ):
            raise ValueError(
                "hidden must have trailing shape (num_assets, d_model)"
            )
        mu_hat = self.return_head(hidden).squeeze(-1)
        return mu_hat

    def normalize_signal(
        self, raw_signal: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Put cross-sectional alpha on the optimizer's quadratic-risk scale.

        The budget constraint makes the cross-sectional level of alpha
        irrelevant, while an unconstrained neural-output scale can turn the
        QP into a linear corner solution.  We therefore use

        ``mu = c * mean(diag(Sigma + eta I)) * (s - mean(s)) / (std(s) + eps)``.

        Normalization is applied outside ``predict_from_hidden`` because each
        horizon token has its own covariance matrix.
        """

        if self.signal_normalization == "none":
            return raw_signal
        if raw_signal.shape[-1] != self.num_assets:
            raise ValueError("raw_signal must have num_assets in its last dimension")
        if sigma.shape[:-2] != raw_signal.shape[:-1] or sigma.shape[-2:] != (
            self.num_assets,
            self.num_assets,
        ):
            raise ValueError("sigma batch shape must match raw_signal")

        centered = raw_signal - raw_signal.mean(dim=-1, keepdim=True)
        cross_sectional_std = raw_signal.std(
            dim=-1, unbiased=False, keepdim=True
        )
        z = centered / (cross_sectional_std + self.signal_normalization_epsilon)
        risk_scale = sigma.diagonal(dim1=-2, dim2=-1).mean(
            dim=-1, keepdim=True
        ) + self.eta
        risk_scale = risk_scale.clamp_min(self.signal_normalization_epsilon)
        return self.signal_scale * risk_scale * z

    def forward(
        self, log_return_path: torch.Tensor, date_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hidden states and logits with shapes ``(B,H,N,M)/(B,H,N)``."""

        hidden = self.encode_inputs(log_return_path, date_feats)
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
        self.bias_scale = nn.Parameter(torch.full((n_heads,), 0.1))

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
            if decision_bias.shape[1] not in (1, self.n_heads):
                raise ValueError("decision_bias head dimension must be 1 or n_heads")
            scores = scores + self.bias_scale.view(1, -1, 1, 1) * decision_bias
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, num_assets, self.d_model
        )
        output = self.dropout(self.output(output))
        return self.norm(hidden + output)


class PrimalDualBiasEncoder(nn.Module):
    """Encode per-asset primal-dual states into low-rank multi-head bias."""

    def __init__(self, input_dim: int, d_model: int, n_heads: int, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("KKT bias rank must be positive")
        self.n_heads = n_heads
        self.rank = rank
        self.norm = nn.LayerNorm(input_dim)
        self.context = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.bias_query = nn.Linear(input_dim, n_heads * rank, bias=False)
        self.bias_key = nn.Linear(input_dim, n_heads * rank, bias=False)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("KKT features must have shape (B, H, N, K)")
        batch, horizon, assets, _ = features.shape
        z = self.norm(features)
        context = self.context(z)
        q = self.bias_query(z).view(batch, horizon, assets, self.n_heads, self.rank)
        k = self.bias_key(z).view(batch, horizon, assets, self.n_heads, self.rank)
        bias = torch.einsum("bhiar,bhjar->bhaij", q, k) / (self.rank ** 0.5)
        scale = bias.abs().mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return context, (bias / scale).clamp(-5.0, 5.0)


class DecisionAwareModel(nn.Module):
    """Sequence-to-sequence KKTFormer with primal-dual attention feedback."""

    def __init__(self, configs, feedback_mode: str = "dual"):
        super().__init__()
        feedback_mode = str(feedback_mode).lower()
        if feedback_mode not in {"dual", "jacobian"}:
            raise ValueError("feedback_mode must be dual or jacobian")
        self.feedback_mode = feedback_mode
        self.backbone = Model(configs)
        self.num_assets = self.backbone.num_assets
        self.d_model = self.backbone.d_model
        # Keep the feedback token strictly optimizer-derived.  Raw factor
        # exposures are ordinary market features, not KKT information.
        self.kkt_feature_dim = 7
        self.kkt_bias_rank = int(getattr(configs, "kkt_bias_rank", 4))
        self.attention = DecisionAwareAssetAttention(
            d_model=self.d_model,
            n_heads=self.backbone.n_heads,
            dropout=self.backbone.dropout,
        )
        self.refined_norm = nn.LayerNorm(self.d_model)
        self.kkt_encoder = PrimalDualBiasEncoder(
            input_dim=self.kkt_feature_dim,
            d_model=self.d_model,
            n_heads=self.backbone.n_heads,
            rank=self.kkt_bias_rank,
        )

    def initial_forward(
        self, log_return_path: torch.Tensor, date_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone.encode_inputs(log_return_path, date_feats)
        return hidden, self.backbone.predict_from_hidden(hidden)

    def normalize_signal(
        self, raw_signal: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone.normalize_signal(raw_signal, sigma)

    def refine(
        self,
        hidden: torch.Tensor,
        kkt_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape (B, H, N, d_model)")
        weights = kkt_state["weights"]
        marginal_risk = kkt_state["marginal_risk"]
        lower_dual = kkt_state["lower_dual"]
        upper_dual = kkt_state["upper_dual"]
        active_lower = kkt_state["active_lower"].to(hidden.dtype)
        active_upper = kkt_state["active_upper"].to(hidden.dtype)

        kkt_features = torch.cat(
            [
                weights.unsqueeze(-1),
                marginal_risk.unsqueeze(-1),
                lower_dual.unsqueeze(-1),
                upper_dual.unsqueeze(-1),
                active_lower.unsqueeze(-1),
                active_upper.unsqueeze(-1),
                kkt_state["pressure"].unsqueeze(-1),
            ],
            dim=-1,
        )
        decision_context, decision_bias = self.kkt_encoder(kkt_features)
        if self.feedback_mode == "jacobian":
            jacobian = kkt_state["jacobian"]
            jacobian = jacobian / jacobian.abs().mean(
                dim=(-1, -2), keepdim=True
            ).clamp_min(1e-6)
            decision_bias = decision_bias + jacobian.clamp(-5.0, 5.0).unsqueeze(2)

        batch, horizon, assets, width = hidden.shape
        refined_hidden = self.attention(
            hidden.reshape(batch * horizon, assets, width),
            decision_context.reshape(batch * horizon, assets, width),
            decision_bias=decision_bias.reshape(
                batch * horizon, self.backbone.n_heads, assets, assets
            ),
        )
        refined_hidden = self.refined_norm(refined_hidden).reshape(
            batch, horizon, assets, width
        )
        return refined_hidden, self.backbone.predict_from_hidden(refined_hidden)

    def forward(
        self,
        log_return_path: torch.Tensor,
        date_feats: torch.Tensor,
        kkt_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden0, mu0 = self.initial_forward(log_return_path, date_feats)
        hidden1, mu1 = self.refine(hidden0, kkt_state)
        return hidden0, mu0, hidden1, mu1
