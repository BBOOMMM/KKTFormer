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
        self.log_return_embed_dim = int(
            getattr(configs, "log_return_embed_dim", 32)
        )
        self.date_embed_dim = int(
            getattr(configs, "date_embed_dim", 32)
        )
        self.asset_embed_dim = int(
            getattr(configs, "asset_embed_dim", 32)
        )
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
        # Risk statistics are deliberately bounded by the model input window.
        # The path is stored as [zero, log returns], so a 60-point path has 59
        # genuine observations.  Several sub-windows provide a multi-scale
        # prior without importing any history outside the SIT window.
        self.risk_momentum_lookback = int(
            getattr(configs, "risk_momentum_lookback", self.lookback_window)
        )
        self.risk_momentum_short_weight = float(
            getattr(configs, "risk_momentum_short_weight", 0.0)
        )
        self.risk_momentum_residual_weight = float(
            getattr(configs, "risk_momentum_residual_weight", 0.0)
        )
        self.risk_forecast_weight = float(
            getattr(configs, "risk_forecast_weight", 0.0)
        )
        # The 60-point window contains several distinct decision regimes.  A
        # trend-only prior is brittle when a recent rally is exhausted, while
        # a pure low-volatility rule gives up too much return.  The policy
        # therefore learns a convex mixture of trend, short-horizon reversal,
        # and downside-risk signals.  These are decision features, not a
        # prediction target, and are trained only through the portfolio loss.
        self.risk_contrarian_weight = float(
            getattr(configs, "risk_contrarian_weight", 0.35)
        )
        self.risk_defensive_weight = float(
            getattr(configs, "risk_defensive_weight", 0.15)
        )
        self.risk_prior_bias = float(
            getattr(configs, "risk_prior_bias", 0.4)
        )
        self.risk_score_normalization = str(
            getattr(configs, "risk_score_normalization", "zscore")
        ).lower()
        self.risk_score_epsilon = float(
            getattr(configs, "risk_score_epsilon", 1e-4)
        )
        self.risk_multiscale_residual_weight = float(
            getattr(configs, "risk_multiscale_residual_weight", 1.0)
        )
        self.risk_defensive_gate_floor = float(
            getattr(configs, "risk_defensive_gate_floor", 0.0)
        )

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(
            self.log_return_embed_dim,
            self.date_embed_dim,
            self.asset_embed_dim,
        ) <= 0:
            raise ValueError("input embedding dimensions must be positive")
        if self.signal_normalization not in {"risk", "none"}:
            raise ValueError("signal_normalization must be risk or none")
        if self.risk_score_normalization not in {"zscore", "raw"}:
            raise ValueError("risk_score_normalization must be zscore or raw")
        if self.risk_score_epsilon <= 0:
            raise ValueError("risk_score_epsilon must be positive")
        if not 0.0 <= self.risk_multiscale_residual_weight <= 1.0:
            raise ValueError("risk_multiscale_residual_weight must be in [0, 1]")
        if not 0.0 <= self.risk_defensive_gate_floor < 1.0:
            raise ValueError("risk_defensive_gate_floor must be in [0, 1)")
        if self.signal_scale <= 0:
            raise ValueError("signal_scale must be positive")
        if self.signal_normalization_epsilon <= 0:
            raise ValueError("signal_normalization_epsilon must be positive")
        if self.risk_momentum_lookback <= 0:
            raise ValueError("risk_momentum_lookback must be positive")
        if not 0.0 <= self.risk_momentum_short_weight <= 1.0:
            raise ValueError("risk_momentum_short_weight must be in [0, 1]")
        if self.risk_momentum_residual_weight < 0:
            raise ValueError("risk_momentum_residual_weight cannot be negative")
        if self.risk_forecast_weight < 0:
            raise ValueError("risk_forecast_weight cannot be negative")
        if self.risk_contrarian_weight < 0 or self.risk_defensive_weight < 0:
            raise ValueError("risk signal weights cannot be negative")

        raw_scales = getattr(configs, "risk_scale_windows", "20,40,60")
        if isinstance(raw_scales, str):
            scale_values = [item.strip() for item in raw_scales.split(",") if item.strip()]
        else:
            scale_values = list(raw_scales)
        self.risk_scale_windows = tuple(sorted({int(item) for item in scale_values}))
        if not self.risk_scale_windows or any(item <= 0 for item in self.risk_scale_windows):
            raise ValueError("risk_scale_windows must contain positive windows")
        if any(item > self.lookback_window for item in self.risk_scale_windows):
            raise ValueError(
                "risk_scale_windows cannot exceed the model lookback window"
            )

        fusion_input_dim = (
            self.log_return_embed_dim
            + self.date_embed_dim
            + self.asset_embed_dim
        )
        self.path_projection = nn.Linear(
            self.lookback_window, self.log_return_embed_dim
        )
        self.date_projection = nn.Linear(self.time_feat_dim, self.date_embed_dim)
        self.asset_embedding = nn.Embedding(
            self.num_assets, self.asset_embed_dim
        )
        self.concat_projection = nn.Linear(fusion_input_dim, self.d_model)
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
        # self.return_head = nn.Linear(self.d_model, 1, bias=True)
        # Keep allocation logits separate from the risk-scaled return signal.
        # The latter is deliberately tiny and would make a direct softmax
        # nearly uniform; this head is free to learn the concentration scale.
        self.allocation_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )
        # A separate decision residual is zero-initialized.  Risk-budget
        # policies therefore start from the causal prior rather than a random
        # alpha ranking, while the final layer is free to learn from CVaR and
        # decision regret once portfolio gradients arrive.
        # Do not let adding this zero-start branch perturb the initialization
        # of the existing gates under a fixed experiment seed.
        residual_rng_state = torch.get_rng_state()
        self.risk_residual_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )
        nn.init.zeros_(self.risk_residual_head[-1].weight)
        nn.init.zeros_(self.risk_residual_head[-1].bias)
        torch.set_rng_state(residual_rng_state)
        # self.allocation_head = nn.Linear(self.d_model, 1, bias=True)
        gate_width = max(16, self.d_model // 2)
        scale_count = len(self.risk_scale_windows)
        self.risk_scale_gate = nn.Sequential(
            nn.LayerNorm(self.d_model + scale_count),
            nn.Linear(self.d_model + scale_count, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, scale_count),
        )
        self.risk_signal_gate = nn.Sequential(
            nn.LayerNorm(self.d_model + 3),
            nn.Linear(self.d_model + 3, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, 3),
        )
        self.risk_prior_gate = nn.Sequential(
            nn.LayerNorm(self.d_model + 2),
            nn.Linear(self.d_model + 2, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, 1),
        )
        self.risk_forecast_gate = nn.Sequential(
            nn.LayerNorm(self.d_model + 3),
            nn.Linear(self.d_model + 3, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, 1),
        )
        # The gate hidden layers remain expressive, but their output layers
        # start at zero.  Thus every seed begins with the same neutral gate:
        # uniform scale/signal mixing and 0.5 sigmoid weights for prior and
        # forecast routes.  The first optimizer step learns only the output
        # projection; hidden gate parameters become active after that update.
        # This removes an otherwise large source of seed-dependent portfolio
        # rankings at initialization.
        for gate in (
            self.risk_scale_gate,
            self.risk_signal_gate,
            self.risk_prior_gate,
            self.risk_forecast_gate,
        ):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

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

    def allocation_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Produce unconstrained logits for a long-only simplex policy."""

        if hidden.ndim < 3 or hidden.shape[-2:] != (
            self.num_assets,
            self.d_model,
        ):
            raise ValueError(
                "hidden must have trailing shape (num_assets, d_model)"
            )
        return self.allocation_head(hidden).squeeze(-1)

    def risk_budget_logits(
        self,
        log_return_path: torch.Tensor,
        hidden: torch.Tensor,
        kkt_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return multi-scale risk-budget logits with a learned prior gate.

        Every statistic is computed from the supplied causal path.  The model
        learns both which of the 20/40/60-style scales to trust and how much
        to mix the risk prior with the Transformer allocation residual.
        """

        if log_return_path.ndim != 4:
            raise ValueError("log_return_path must have shape (B, H, N, W)")
        if hidden.shape[:-1] != log_return_path.shape[:-1]:
            raise ValueError("hidden and log_return_path have incompatible shapes")
        # The leading zero is a representation sentinel, not a return.
        returns = torch.expm1(log_return_path[..., 1:])
        available = returns.shape[-1]
        max_window = min(self.risk_momentum_lookback, available)

        def raw_score(window):
            return window.sum(dim=-1) / window.std(
                dim=-1, unbiased=False
            ).add(self.risk_score_epsilon)

        def score(window):
            value = raw_score(window)
            centered = value - value.mean(dim=-1, keepdim=True)
            return centered / value.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(self.signal_normalization_epsilon)

        scale_scores = []
        scale_raw_scores = []
        for requested in self.risk_scale_windows:
            scale = min(requested, max_window)
            scale_scores.append(score(returns[..., -scale:]))
            scale_raw_scores.append(raw_score(returns[..., -scale:]))
        scores = torch.stack(scale_scores, dim=-1)
        raw_scores = torch.stack(scale_raw_scores, dim=-1)
        scale_gate_input = torch.cat((hidden, scores), dim=-1)
        scale_weights = torch.softmax(self.risk_scale_gate(scale_gate_input), dim=-1)
        if self.risk_score_normalization == "raw":
            mixed_trend = (scale_weights * raw_scores).sum(dim=-1)
            base_trend = raw_scores[..., 0]
            trend = base_trend + self.risk_multiscale_residual_weight * (
                mixed_trend - base_trend
            )
        else:
            trend = (scale_weights * scores).sum(dim=-1)
            trend = trend - trend.mean(dim=-1, keepdim=True)
            trend = trend / trend.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(self.signal_normalization_epsilon)

        # Short-term reversal is computed from the same causal 60-point path.
        # It complements, rather than replaces, the multi-scale trend route.
        short_scale = min(max(5, max_window // 4), available)
        short_returns = returns[..., -short_scale:]
        short_trend = score(short_returns)
        reversal = (
            -raw_score(short_returns)
            if self.risk_score_normalization == "raw"
            else -score(short_returns)
        )
        # Prefer assets with a smaller recent downside semivariance when the
        # learned gate enters a defensive regime.
        # Defensive exposure is tactical rather than a second long-horizon
        # trend estimate.  Keeping it on the 20-day decision scale avoids
        # mixing a slow 59-day volatility regime into the raw risk temperature.
        defensive_window = min(20, max_window)
        downside = torch.sqrt(
            torch.mean(
                torch.relu(-returns[..., -defensive_window:]).square(), dim=-1
            )
        )
        defensive = -downside
        defensive = defensive - defensive.mean(dim=-1, keepdim=True)
        defensive = defensive / defensive.std(
            dim=-1, unbiased=False, keepdim=True
        ).clamp_min(self.signal_normalization_epsilon)
        signal_features = torch.stack((trend, reversal, defensive), dim=-1)
        signal_gate_input = torch.cat((hidden, signal_features), dim=-1)
        signal_weights = torch.softmax(
            self.risk_signal_gate(signal_gate_input), dim=-1
        )
        # The trend route is the identifiable base policy.  The gates only
        # control the strength of the complementary reversal/defensive
        # corrections; multiplying the trend by its gate would make the
        # portfolio temperature depend on an arbitrary hidden-state gate.
        # Keep the defensive gate active in raw-score mode as well: raw mode
        # changes score geometry, not whether the Transformer can route the
        # downside signal.
        defensive_gate = self.risk_defensive_gate_floor + (
            1.0 - self.risk_defensive_gate_floor
        ) * signal_weights[..., 2]
        prior = (
            trend
            + self.risk_momentum_short_weight * short_trend
            + self.risk_contrarian_weight * signal_weights[..., 1] * reversal
            + self.risk_defensive_weight * defensive_gate * defensive
        )
        if self.risk_score_normalization != "raw":
            prior = prior - prior.mean(dim=-1, keepdim=True)
            prior = prior / prior.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(self.signal_normalization_epsilon)

        learned_residual = self.risk_residual_head(hidden).squeeze(-1)
        residual = learned_residual - learned_residual.mean(
            dim=-1, keepdim=True
        )
        residual = residual / residual.std(
            dim=-1, unbiased=False, keepdim=True
        ).clamp_min(self.signal_normalization_epsilon)
        # Apply the coefficient after normalization.  Applying it before the
        # z-score would cancel the coefficient and turn any nonzero value into
        # an uncontrolled full-strength residual policy.
        residual = self.risk_momentum_residual_weight * residual
        gate_features = torch.cat(
            (hidden, prior.unsqueeze(-1), scores.mean(dim=-1, keepdim=True)), dim=-1
        )
        prior_weight = torch.sigmoid(self.risk_prior_gate(gate_features)).squeeze(-1)
        forecast = self.predict_from_hidden(hidden)
        forecast = forecast - forecast.mean(dim=-1, keepdim=True)
        forecast = forecast / forecast.std(
            dim=-1, unbiased=False, keepdim=True
        ).clamp_min(self.signal_normalization_epsilon)
        forecast_features = torch.cat(
            (
                hidden,
                prior.unsqueeze(-1),
                residual.unsqueeze(-1),
                forecast.unsqueeze(-1),
            ),
            dim=-1,
        )
        forecast_gate = torch.sigmoid(
            self.risk_forecast_gate(forecast_features)
        ).squeeze(-1)
        if self.risk_momentum_residual_weight == 0.0:
            blended = prior
        else:
            # The learned allocation head is a residual policy around the
            # causal risk prior.  The prior gate controls residual injection,
            # but never removes the causal baseline entirely:
            #   blended = prior + (1 - gate) * residual.
            # This keeps the decision signal identifiable while ensuring the
            # Transformer representation receives a real portfolio gradient.
            blended = prior_weight * prior + (1.0 - prior_weight) * (
                prior + residual
            )
        return blended + self.risk_forecast_weight * forecast_gate * forecast

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
        self.dynamic_query = nn.Linear(d_model, d_model, bias=False)
        self.dynamic_key = nn.Linear(d_model, d_model, bias=False)
        self.dynamic_gate_raw = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        hidden: torch.Tensor,
        decision_context: torch.Tensor,
        decision_bias: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        dynamic_bias: bool = False,
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
        if dynamic_bias:
            if relation_context is None:
                raise ValueError("relation_context is required for dynamic_bias")
            relation_q = self.dynamic_query(hidden).view(
                batch_size, num_assets, self.n_heads, self.head_dim
            ).transpose(1, 2)
            relation_k = self.dynamic_key(relation_context).view(
                batch_size, num_assets, self.n_heads, self.head_dim
            ).transpose(1, 2)
            dynamic = torch.matmul(
                relation_q, relation_k.transpose(-1, -2)
            ) / (self.head_dim ** 0.5)
            dynamic = dynamic / dynamic.detach().abs().mean(
                dim=(-1, -2), keepdim=True
            ).clamp_min(1e-6)
            scores = scores + torch.tanh(self.dynamic_gate_raw).view(
                1, -1, 1, 1
            ) * dynamic.clamp(-5.0, 5.0)
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
        if feedback_mode not in {
            "two_pass",
            "context",
            "bias",
            "dynamic",
            "dual",
            "jacobian",
        }:
            raise ValueError(
                "feedback_mode must be one of two_pass, context, bias, dual, "
                "or jacobian"
            )
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
        self.kkt_risk_scale = float(getattr(configs, "kkt_risk_scale", 0.1))
        if self.kkt_risk_scale < 0:
            raise ValueError("kkt_risk_scale cannot be negative")
        # This gate is deliberately separate from the attention gate.  It
        # lets the optimizer state influence the final risk-budget decision
        # even when the KKT representation is not sufficiently expressive to
        # improve the intermediate asset-attention hidden state.
        gate_width = max(8, self.d_model // 2)
        self.kkt_risk_gate = nn.Sequential(
            nn.LayerNorm(self.kkt_feature_dim),
            nn.Linear(self.kkt_feature_dim, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, 1),
        )
        # Keep the KKT gate deterministic at initialization as well.  A zero
        # sigmoid logit gives a fixed 0.5 injection gate; its asset-dependent
        # modulation is learned only after the output layer receives a
        # gradient.
        nn.init.zeros_(self.kkt_risk_gate[-1].weight)
        nn.init.zeros_(self.kkt_risk_gate[-1].bias)

    def initial_forward(
        self, log_return_path: torch.Tensor, date_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone.encode_inputs(log_return_path, date_feats)
        return hidden, self.backbone.predict_from_hidden(hidden)

    def normalize_signal(
        self, raw_signal: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone.normalize_signal(raw_signal, sigma)

    def allocation_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.backbone.allocation_logits_from_hidden(hidden)

    def risk_budget_logits(
        self,
        log_return_path: torch.Tensor,
        hidden: torch.Tensor,
        kkt_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        logits = self.backbone.risk_budget_logits(log_return_path, hidden)
        if kkt_state is None or self.kkt_risk_scale == 0.0:
            return logits

        required = {
            "weights",
            "marginal_risk",
            "lower_dual",
            "upper_dual",
            "active_lower",
            "active_upper",
            "pressure",
        }
        missing = required.difference(kkt_state)
        if missing:
            raise ValueError(f"KKT state is missing risk features: {sorted(missing)}")
        features = torch.cat(
            [
                kkt_state["weights"].unsqueeze(-1),
                kkt_state["marginal_risk"].unsqueeze(-1),
                kkt_state["lower_dual"].unsqueeze(-1),
                kkt_state["upper_dual"].unsqueeze(-1),
                kkt_state["active_lower"].to(logits.dtype).unsqueeze(-1),
                kkt_state["active_upper"].to(logits.dtype).unsqueeze(-1),
                kkt_state["pressure"].unsqueeze(-1),
            ],
            dim=-1,
        )

        def cross_sectional_zscore(value):
            centered = value - value.mean(dim=-1, keepdim=True)
            return centered / value.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(1e-6)

        # A signed KKT risk score: avoid high marginal risk and lower-bound
        # pressure, while retaining assets whose upper-bound multiplier says
        # that the alpha signal is economically valuable.  The learned gate
        # controls how much of this score is trusted for each token.
        kkt_signal = (
            -cross_sectional_zscore(kkt_state["marginal_risk"])
            + 0.5 * cross_sectional_zscore(kkt_state["upper_dual"])
            - 0.5 * cross_sectional_zscore(kkt_state["lower_dual"])
            - 0.25 * cross_sectional_zscore(kkt_state["pressure"])
        )
        gate = torch.sigmoid(self.kkt_risk_gate(features).squeeze(-1))
        return logits + self.kkt_risk_scale * gate * kkt_signal

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
        relation_context = decision_context
        # Keep every two-pass ablation parameter- and architecture-matched.
        # Only the information supplied to the refinement attention changes:
        #   two_pass: neither KKT path; context: hidden/context path only;
        #   bias: attention-score path only; dual: both KKT paths.
        if self.feedback_mode in {"two_pass", "bias", "dynamic"}:
            decision_context = torch.zeros_like(decision_context)
        if self.feedback_mode in {"two_pass", "context", "dynamic"}:
            decision_bias = torch.zeros_like(decision_bias)
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
            relation_context=relation_context.reshape(
                batch * horizon, assets, width
            ) if self.feedback_mode == "dynamic" else None,
            dynamic_bias=self.feedback_mode == "dynamic",
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
