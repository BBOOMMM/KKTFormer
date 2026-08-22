"""KKTFormer with SIT-aligned path, date, and asset token inputs."""

import math
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
        self.asset_embedding_scale = float(
            getattr(configs, "asset_embedding_scale", 1.0)
        )
        self.asset_embedding_init = str(
            getattr(configs, "asset_embedding_init", "random")
        ).lower()
        self.portfolio_heads = int(getattr(configs, "portfolio_heads", 1))
        self.portfolio_aggregation = str(
            getattr(configs, "portfolio_aggregation", "probability_mean")
        ).lower()
        self.spectral_policy_filters = int(
            getattr(configs, "spectral_policy_filters", 0)
        )
        self.spectral_policy_hidden = int(
            getattr(configs, "spectral_policy_hidden", max(8, self.d_model // 2))
        )
        self.spectral_policy_scale = float(
            getattr(configs, "spectral_policy_scale", 1.0)
        )
        self.tail_policy_filters = int(
            getattr(configs, "tail_policy_filters", 0)
        )
        self.tail_policy_hidden = int(
            getattr(configs, "tail_policy_hidden", max(8, self.d_model // 2))
        )
        self.tail_policy_scale = float(
            getattr(configs, "tail_policy_scale", 1.0)
        )
        self.ordered_policy_bins = int(
            getattr(configs, "ordered_policy_bins", 8)
        )
        self.ordered_policy_scale = float(
            getattr(configs, "ordered_policy_scale", 0.0)
        )
        self.policy_refinement_steps = int(
            getattr(configs, "policy_refinement_steps", 0)
        )
        self.policy_refinement_scale = float(
            getattr(configs, "policy_refinement_scale", 0.0)
        )
        self.policy_refinement_window = int(
            getattr(configs, "policy_refinement_window", self.lookback_window - 1)
        )
        self.policy_refinement_risk = str(
            getattr(configs, "policy_refinement_risk", "variance")
        ).lower()
        self.policy_refinement_tail_temperature = float(
            getattr(configs, "policy_refinement_tail_temperature", 1e-2)
        )
        self.lpm_geometry_scale = float(
            getattr(configs, "lpm_geometry_scale", 0.0)
        )
        self.lpm_geometry_window = int(
            getattr(configs, "lpm_geometry_window", 10)
        )
        self.lpm_geometry_init = float(
            getattr(configs, "lpm_geometry_init", 2.5)
        )
        self.lpm_geometry_epsilon = float(
            getattr(configs, "lpm_geometry_epsilon", 1e-4)
        )
        self.relation_attention_scale = float(
            getattr(configs, "relation_attention_scale", 0.0)
        )
        self.relation_attention_hidden = int(
            getattr(configs, "relation_attention_hidden", 16)
        )
        raw_tail_windows = getattr(configs, "tail_policy_windows", "5,10,20,60")
        if isinstance(raw_tail_windows, str):
            tail_windows = [
                item.strip() for item in raw_tail_windows.split(",") if item.strip()
            ]
        else:
            tail_windows = list(raw_tail_windows)
        genuine_window = max(1, self.lookback_window - 1)
        self.tail_policy_windows = tuple(
            sorted({min(int(item), genuine_window) for item in tail_windows})
        )
        raw_ordered_windows = getattr(
            configs, "ordered_policy_windows", "5,10,20,60"
        )
        if isinstance(raw_ordered_windows, str):
            ordered_windows = [
                item.strip()
                for item in raw_ordered_windows.split(",")
                if item.strip()
            ]
        else:
            ordered_windows = list(raw_ordered_windows)
        self.ordered_policy_windows = tuple(
            sorted({min(int(item), genuine_window) for item in ordered_windows})
        )
        self.use_asset_policy_bias = bool(
            int(getattr(configs, "use_asset_policy_bias", 0))
        )
        self.asset_policy_bias_scale = float(
            getattr(configs, "asset_policy_bias_scale", 1.0)
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
        self.risk_gate_logit_scale = float(
            getattr(configs, "risk_gate_logit_scale", 1.0)
        )
        self.risk_gate_init = str(
            getattr(configs, "risk_gate_init", "neutral")
        ).lower()
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
        if not math.isfinite(self.asset_embedding_scale) or self.asset_embedding_scale < 0:
            raise ValueError("asset_embedding_scale must be finite and non-negative")
        if self.asset_embedding_init not in {"random", "deterministic", "orthogonal"}:
            raise ValueError(
                "asset_embedding_init must be random, deterministic, or orthogonal"
            )
        if self.portfolio_heads <= 0:
            raise ValueError("portfolio_heads must be positive")
        if self.portfolio_aggregation not in {"probability_mean", "logit_mean"}:
            raise ValueError(
                "portfolio_aggregation must be probability_mean or logit_mean"
            )
        if self.spectral_policy_filters < 0:
            raise ValueError("spectral_policy_filters cannot be negative")
        if self.spectral_policy_filters > 0 and self.spectral_policy_hidden < 0:
            raise ValueError("spectral_policy_hidden cannot be negative")
        if (
            not math.isfinite(self.spectral_policy_scale)
            or self.spectral_policy_scale < 0
        ):
            raise ValueError("spectral_policy_scale must be finite and non-negative")
        if self.tail_policy_filters < 0:
            raise ValueError("tail_policy_filters cannot be negative")
        if self.tail_policy_filters > 0 and self.tail_policy_hidden < 0:
            raise ValueError("tail_policy_hidden cannot be negative")
        if not math.isfinite(self.tail_policy_scale) or self.tail_policy_scale < 0:
            raise ValueError("tail_policy_scale must be finite and non-negative")
        if self.ordered_policy_bins <= 0:
            raise ValueError("ordered_policy_bins must be positive")
        if (
            not math.isfinite(self.ordered_policy_scale)
            or self.ordered_policy_scale < 0
        ):
            raise ValueError(
                "ordered_policy_scale must be finite and non-negative"
            )
        if self.policy_refinement_steps < 0:
            raise ValueError("policy_refinement_steps cannot be negative")
        if (
            not math.isfinite(self.policy_refinement_scale)
            or self.policy_refinement_scale < 0
        ):
            raise ValueError(
                "policy_refinement_scale must be finite and non-negative"
            )
        if self.policy_refinement_window <= 1:
            raise ValueError("policy_refinement_window must be greater than one")
        if self.policy_refinement_risk not in {"variance", "cvar", "lpm"}:
            raise ValueError(
                "policy_refinement_risk must be variance, cvar, or lpm"
            )
        if (
            not math.isfinite(self.policy_refinement_tail_temperature)
            or self.policy_refinement_tail_temperature <= 0.0
        ):
            raise ValueError(
                "policy_refinement_tail_temperature must be finite and positive"
            )
        if (
            not math.isfinite(self.lpm_geometry_scale)
            or self.lpm_geometry_scale < 0.0
        ):
            raise ValueError("lpm_geometry_scale must be finite and non-negative")
        if self.lpm_geometry_window <= 1:
            raise ValueError("lpm_geometry_window must be greater than one")
        if (
            not math.isfinite(self.lpm_geometry_init)
            or self.lpm_geometry_init <= 0.0
        ):
            raise ValueError("lpm_geometry_init must be finite and positive")
        if (
            not math.isfinite(self.lpm_geometry_epsilon)
            or self.lpm_geometry_epsilon <= 0.0
        ):
            raise ValueError("lpm_geometry_epsilon must be finite and positive")
        if (
            not math.isfinite(self.relation_attention_scale)
            or self.relation_attention_scale < 0
        ):
            raise ValueError(
                "relation_attention_scale must be finite and non-negative"
            )
        if self.relation_attention_hidden <= 0:
            raise ValueError("relation_attention_hidden must be positive")
        if not self.tail_policy_windows or any(
            item <= 0 for item in self.tail_policy_windows
        ):
            raise ValueError("tail_policy_windows must contain positive windows")
        if not self.ordered_policy_windows or any(
            item <= 0 for item in self.ordered_policy_windows
        ):
            raise ValueError("ordered_policy_windows must contain positive windows")
        if (
            not math.isfinite(self.asset_policy_bias_scale)
            or self.asset_policy_bias_scale < 0
        ):
            raise ValueError(
                "asset_policy_bias_scale must be finite and non-negative"
            )
        if self.risk_gate_logit_scale < 0:
            raise ValueError("risk_gate_logit_scale cannot be negative")
        if self.risk_gate_init not in {"neutral", "decision"}:
            raise ValueError("risk_gate_init must be neutral or decision")
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
        # Short synthetic smoke-test windows (and small live universes) may
        # be narrower than the default 20/40/60-day route list.  Clipping the
        # requested scales to the available causal path preserves the route
        # semantics and avoids making model construction depend on a test-only
        # configuration detail.
        self.risk_scale_windows = tuple(
            sorted({min(item, self.lookback_window) for item in self.risk_scale_windows})
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
        if self.asset_embedding_init == "orthogonal":
            # A categorical identity basis avoids imposing an arbitrary
            # similarity ordering on assets. With enough channels it is an
            # exact one-hot code; narrower ablations use orthogonal cosine
            # columns. The embedding remains trainable.
            row = torch.arange(self.num_assets, dtype=self.asset_embedding.weight.dtype)
            column = torch.arange(
                self.asset_embed_dim, dtype=self.asset_embedding.weight.dtype
            )
            if self.asset_embed_dim >= self.num_assets:
                codes = torch.zeros_like(self.asset_embedding.weight)
                codes[:, : self.num_assets] = torch.eye(
                    self.num_assets, dtype=codes.dtype
                )
                codes = codes * math.sqrt(float(self.asset_embed_dim))
            else:
                codes = torch.cos(
                    math.pi
                    * (row.unsqueeze(1) + 0.5)
                    * column.unsqueeze(0)
                    / float(self.num_assets)
                )
                codes = codes / codes.square().mean(
                    dim=-1, keepdim=True
                ).sqrt().clamp_min(1e-6)
            with torch.no_grad():
                self.asset_embedding.weight.copy_(codes)
        elif self.asset_embedding_init == "deterministic":
            # Asset identity is categorical, so the code deliberately avoids
            # an ordinal ramp. Irrational Fourier phases give every asset a
            # distinct, bounded starting code while remaining identical across
            # experiment seeds. The table remains fully trainable afterwards.
            asset_index = torch.arange(
                1, self.num_assets + 1, dtype=self.asset_embedding.weight.dtype
            ).unsqueeze(1)
            channel_index = torch.arange(
                1, self.asset_embed_dim + 1,
                dtype=self.asset_embedding.weight.dtype,
            ).unsqueeze(0)
            codes = torch.sin(math.sqrt(2.0) * asset_index * channel_index)
            codes = codes + torch.cos(
                math.sqrt(3.0) * asset_index * (channel_index + 0.5)
            )
            codes = codes / codes.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
                1e-6
            )
            with torch.no_grad():
                self.asset_embedding.weight.copy_(codes)
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

        # Causal path relations condition asset-attention scores rather than
        # directly constructing positions. Signed correlation, correlation
        # magnitude, and downside co-movement are transformed by a learned
        # per-head kernel trained only through realized portfolio outcomes.
        relation_rng_state = torch.get_rng_state()
        self.relation_attention_head = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.relation_attention_hidden),
            nn.GELU(),
            nn.Linear(self.relation_attention_hidden, self.n_heads),
        )
        nn.init.zeros_(self.relation_attention_head[-1].weight)
        nn.init.zeros_(self.relation_attention_head[-1].bias)
        self.relation_attention_gain = nn.Parameter(torch.zeros(self.n_heads))
        # A disabled relation route must not perturb the common backbone.
        torch.set_rng_state(relation_rng_state)

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
        # Start the differentiable policy from a neutral cross-sectional
        # signal.  The head still receives gradients from CVaR/decision
        # objectives, but random initial logits no longer create a different
        # KKT probe and portfolio ranking for every seed.
        nn.init.zeros_(self.return_head[-1].weight)
        nn.init.zeros_(self.return_head[-1].bias)
        # self.return_head = nn.Linear(self.d_model, 1, bias=True)
        # Keep allocation logits separate from the risk-scaled return signal.
        # The latter is deliberately tiny and would make a direct softmax
        # nearly uniform; this head is free to learn the concentration scale.
        # Independent decision experts share the Transformer representation
        # but not their allocation projections. Each expert produces a full
        # learned simplex portfolio and the policy averages those portfolios,
        # rather than shrinking toward an externally specified allocation.
        self.allocation_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model, 1),
                )
                for _ in range(self.portfolio_heads)
            ]
        )
        for head in self.allocation_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if self.use_asset_policy_bias:
            self.asset_policy_bias = nn.Parameter(
                torch.zeros(self.portfolio_heads, self.num_assets)
            )
        else:
            self.register_parameter("asset_policy_bias", None)
        # Optional representation branches must not change the initialization
        # of the common Transformer/KKT policy that follows them.  This keeps
        # architecture-matched 30/40 ablations comparable even when a branch's
        # strength is zero.
        policy_feature_rng_state = torch.get_rng_state()
        if self.spectral_policy_filters > 0:
            # A trainable DCT bank supplies a deterministic, asset-shared
            # temporal coordinate system. It is a generic path representation,
            # not a momentum/risk allocation: the zero-start decision MLP must
            # learn all portfolio logits from the portfolio objective.
            genuine_window = self.lookback_window - 1
            time_index = torch.arange(genuine_window, dtype=torch.float32) + 0.5
            frequency = torch.arange(
                self.spectral_policy_filters, dtype=torch.float32
            ).unsqueeze(1)
            basis = torch.cos(
                math.pi * frequency * time_index.unsqueeze(0) / genuine_window
            )
            basis = basis / basis.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(
                1e-6
            )
            self.spectral_policy_bank = nn.Parameter(basis)
            if self.spectral_policy_hidden == 0:
                self.spectral_policy_head = nn.Sequential(
                    nn.LayerNorm(self.spectral_policy_filters),
                    nn.Linear(self.spectral_policy_filters, self.portfolio_heads),
                )
            else:
                self.spectral_policy_head = nn.Sequential(
                    nn.LayerNorm(self.spectral_policy_filters),
                    nn.Linear(
                        self.spectral_policy_filters, self.spectral_policy_hidden
                    ),
                    nn.GELU(),
                    nn.Linear(self.spectral_policy_hidden, self.portfolio_heads),
                )
            nn.init.zeros_(self.spectral_policy_head[-1].weight)
            nn.init.zeros_(self.spectral_policy_head[-1].bias)
        else:
            self.register_parameter("spectral_policy_bank", None)
            self.spectral_policy_head = None
        if self.tail_policy_filters > 0:
            # Learnable distributional pooling exposes the whole empirical
            # return distribution to the end-to-end policy. Negative queries
            # initially inspect adverse observations, positive queries inspect
            # upside observations, and every query is subsequently learned
            # only through the realized portfolio objective. This is a feature
            # encoder, not a target portfolio or a hand-crafted allocation.
            queries = torch.linspace(-3.0, 3.0, self.tail_policy_filters)
            self.tail_policy_queries = nn.Parameter(queries)
            tail_feature_dim = (
                2 * self.tail_policy_filters * len(self.tail_policy_windows)
            )
            if self.tail_policy_hidden == 0:
                self.tail_policy_head = nn.Sequential(
                    nn.Identity(),
                    nn.Linear(tail_feature_dim, self.portfolio_heads),
                )
            else:
                self.tail_policy_head = nn.Sequential(
                    nn.Identity(),
                    nn.Linear(tail_feature_dim, self.tail_policy_hidden),
                    nn.GELU(),
                    nn.Linear(self.tail_policy_hidden, self.portfolio_heads),
                )
            nn.init.zeros_(self.tail_policy_head[-1].weight)
            nn.init.zeros_(self.tail_policy_head[-1].bias)
        else:
            self.register_parameter("tail_policy_queries", None)
            self.tail_policy_head = None
        # The ordered-distribution branch is shared by every pool size.  Its
        # zero-initialized kernel learns an unrestricted signed weighting over
        # empirical return quantiles and lookback scales.  No quantile is
        # labelled as desirable, no target portfolio is supplied, and the
        # branch is optimized exclusively through the realized portfolio
        # objective.  This gives the softmax policy a low-variance path
        # distribution coordinate without encoding an inverse-risk rule.
        ordered_feature_dim = (
            self.ordered_policy_bins * len(self.ordered_policy_windows)
        )
        self.ordered_policy_head = nn.Linear(
            ordered_feature_dim, self.portfolio_heads, bias=False
        )
        nn.init.zeros_(self.ordered_policy_head.weight)
        # A learned mirror-descent coefficient lets the policy decide how
        # much causal covariance geometry should refine its own logits.  The
        # signed gain starts at zero, so this module is neither a fixed
        # minimum-variance allocation nor an external portfolio anchor.
        self.policy_refinement_gain = nn.Parameter(
            torch.zeros(self.portfolio_heads)
        )
        # Positive risk aversion of the causal LPM geometry.  This scalar is
        # optimized only through the realized portfolio objective.  The
        # inverse-softplus initialization makes the user-facing value the
        # actual initial power while avoiding a positivity projection.
        lpm_raw_init = math.log(math.expm1(self.lpm_geometry_init))
        self.lpm_geometry_raw_power = nn.Parameter(
            torch.tensor(lpm_raw_init, dtype=torch.float32)
        )
        torch.set_rng_state(policy_feature_rng_state)
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
        # Both initializations are deterministic across experiment seeds.  The
        # neutral option is appropriate for the direct softmax policy; the
        # decision warm start retains the validated multi-scale KKT policy's
        # initial geometry while all gates remain end-to-end trainable.
        if self.risk_gate_init == "neutral":
            for gate in (
                self.risk_scale_gate,
                self.risk_signal_gate,
                self.risk_prior_gate,
                self.risk_forecast_gate,
            ):
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
        else:
            with torch.no_grad():
                minimum_window = min(self.risk_scale_windows)
                self.risk_scale_gate[-1].bias.copy_(
                    torch.tensor(
                        [
                            -0.06 * float(window - minimum_window)
                            for window in self.risk_scale_windows
                        ]
                    )
                )
                self.risk_signal_gate[-1].bias.copy_(
                    torch.tensor([1.0, -0.25, -1.0])
                )
                self.risk_prior_gate[-1].bias.fill_(self.risk_prior_bias)
                self.risk_forecast_gate[-1].bias.fill_(-0.5)

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
        asset_emb = (
            self.asset_embedding_scale * self.asset_embedding(asset_ids)
        ).view(1, 1, num_assets, -1).expand(batch_size, horizon, -1, -1)
        x = self.input_norm(self.concat_projection(torch.cat((path_emb, date_emb, asset_emb), dim=-1)))

        # Temporal attention is causal and is applied independently per asset.
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_assets, horizon, self.d_model)
        causal_mask = torch.triu(
            torch.ones(horizon, horizon, device=x.device, dtype=torch.bool), diagonal=1
        )
        x = self.temporal_encoder(x, mask=causal_mask)
        x = x.reshape(batch_size, num_assets, horizon, self.d_model).permute(0, 2, 1, 3)

        # A learned relational message-passing step conditions the token
        # representation before ordinary Transformer asset attention. The
        # zero-start gain keeps the unconditioned backbone as the exact
        # initialization, while avoiding unstable gradients through a
        # sample-specific PyTorch attention mask.
        if self.relation_attention_scale > 0.0:
            returns = torch.expm1(log_return_path[..., 1:])
            centered = returns - returns.mean(dim=-1, keepdim=True)
            normalized = centered / centered.square().sum(
                dim=-1, keepdim=True
            ).clamp_min(self.signal_normalization_epsilon**2).sqrt()
            correlation = torch.einsum(
                "...it,...jt->...ij", normalized, normalized
            ).clamp(-1.0, 1.0)
            downside = torch.relu(-returns)
            downside_normalized = downside / downside.square().sum(
                dim=-1, keepdim=True
            ).clamp_min(self.signal_normalization_epsilon**2).sqrt()
            downside_comovement = torch.einsum(
                "...it,...jt->...ij",
                downside_normalized,
                downside_normalized,
            ).clamp(0.0, 1.0)
            relation_features = torch.stack(
                (correlation, correlation.abs(), downside_comovement), dim=-1
            )
            relation_scores = self.relation_attention_head(
                relation_features
            ).movedim(-1, -3)
            relation_weights = torch.softmax(relation_scores, dim=-1)
            head_dim = self.d_model // self.n_heads
            x_heads = x.reshape(
                batch_size, horizon, num_assets, self.n_heads, head_dim
            ).permute(0, 1, 3, 2, 4)
            relation_context = torch.einsum(
                "btkij,btkjd->btkid", relation_weights, x_heads
            )
            gain = torch.tanh(self.relation_attention_gain).view(
                1, 1, self.n_heads, 1, 1
            )
            x = (
                x_heads
                + self.relation_attention_scale
                * gain
                * (relation_context - x_heads)
            ).permute(0, 1, 3, 2, 4).reshape(
                batch_size, horizon, num_assets, self.d_model
            )

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
        return self.allocation_head_logits_from_hidden(hidden).mean(dim=-2)

    def allocation_head_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        log_return_path: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return independent policy-expert logits as ``(..., K, N)``."""

        if hidden.ndim < 3 or hidden.shape[-2:] != (
            self.num_assets,
            self.d_model,
        ):
            raise ValueError(
                "hidden must have trailing shape (num_assets, d_model)"
            )
        logits = torch.stack(
            [head(hidden).squeeze(-1) for head in self.allocation_heads],
            dim=-2,
        )
        if self.asset_policy_bias is not None:
            logits = logits + self.asset_policy_bias_scale * self.asset_policy_bias
        if self.spectral_policy_head is not None:
            if log_return_path is None:
                raise ValueError(
                    "log_return_path is required when spectral policy is enabled"
                )
            if log_return_path.shape[:-1] != hidden.shape[:-1] or (
                log_return_path.shape[-1] != self.lookback_window
            ):
                raise ValueError("log_return_path and hidden have incompatible shapes")
            spectral_features = torch.einsum(
                "...w,fw->...f",
                log_return_path[..., 1:],
                self.spectral_policy_bank,
            )
            spectral_logits = self.spectral_policy_head(spectral_features).movedim(
                -1, -2
            )
            logits = logits + self.spectral_policy_scale * spectral_logits
        if self.tail_policy_head is not None:
            if log_return_path is None:
                raise ValueError(
                    "log_return_path is required when tail policy is enabled"
                )
            if log_return_path.shape[:-1] != hidden.shape[:-1] or (
                log_return_path.shape[-1] != self.lookback_window
            ):
                raise ValueError("log_return_path and hidden have incompatible shapes")
            all_returns = torch.expm1(log_return_path[..., 1:])
            distribution_features = []
            for window in self.tail_policy_windows:
                returns = all_returns[..., -window:]
                centered = returns - returns.mean(dim=-1, keepdim=True)
                standardized = centered / returns.std(
                    dim=-1, unbiased=False, keepdim=True
                ).clamp_min(self.signal_normalization_epsilon)
                attention = torch.softmax(
                    standardized.unsqueeze(-2)
                    * self.tail_policy_queries.view(
                        *((1,) * (standardized.ndim - 1)), -1, 1
                    ),
                    dim=-1,
                )
                pooled_return = (attention * returns.unsqueeze(-2)).sum(dim=-1)
                pooled_scale = (
                    attention * centered.unsqueeze(-2).square()
                ).sum(dim=-1).clamp_min(1e-12).sqrt()
                distribution_features.extend((pooled_return, pooled_scale))
            tail_features = torch.cat(distribution_features, dim=-1)
            # Normalize each learned distribution coordinate across assets,
            # not across coordinates within one asset. This preserves the
            # relative magnitude of downside dispersion that a per-asset
            # LayerNorm would erase, while remaining permutation equivariant
            # and free of any target allocation.
            tail_features = tail_features - tail_features.mean(
                dim=-2, keepdim=True
            )
            tail_features = tail_features / tail_features.std(
                dim=-2, unbiased=False, keepdim=True
            ).clamp_min(self.signal_normalization_epsilon)
            tail_logits = self.tail_policy_head(tail_features).movedim(-1, -2)
            logits = logits + self.tail_policy_scale * tail_logits
        if self.ordered_policy_scale > 0.0:
            if log_return_path is None:
                raise ValueError(
                    "log_return_path is required when ordered policy is enabled"
                )
            if log_return_path.shape[:-1] != hidden.shape[:-1] or (
                log_return_path.shape[-1] != self.lookback_window
            ):
                raise ValueError("log_return_path and hidden have incompatible shapes")
            all_returns = torch.expm1(log_return_path[..., 1:])
            ordered_features = []
            for window in self.ordered_policy_windows:
                ordered = all_returns[..., -window:].sort(dim=-1).values
                # Fixed evenly spaced ranks define only a coordinate system;
                # their signed importance is fully learned.  Rounding keeps
                # every selected value an exact order statistic and therefore
                # differentiable with respect to the observed return.
                rank_index = torch.linspace(
                    0,
                    window - 1,
                    self.ordered_policy_bins,
                    device=ordered.device,
                ).round().long()
                quantiles = ordered.index_select(-1, rank_index)
                quantiles = quantiles - quantiles.mean(dim=-2, keepdim=True)
                quantiles = quantiles / quantiles.std(
                    dim=-2, unbiased=False, keepdim=True
                ).clamp_min(self.signal_normalization_epsilon)
                ordered_features.append(quantiles)
            ordered_features = torch.cat(ordered_features, dim=-1)
            ordered_logits = self.ordered_policy_head(
                ordered_features
            ).movedim(-1, -2)
            # Dimension normalization makes the residual scale comparable
            # when the number of quantile coordinates is changed.
            ordered_logits = ordered_logits / math.sqrt(
                float(ordered_features.shape[-1])
            )
            logits = logits + self.ordered_policy_scale * ordered_logits
        return logits

    def allocation_weights_from_hidden(
        self,
        hidden: torch.Tensor,
        temperature: float,
        log_return_path: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average fully learned expert portfolios on the probability simplex."""

        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        head_logits = self.allocation_head_logits_from_hidden(
            hidden, log_return_path=log_return_path
        )
        head_logits = self.refine_policy_logits(
            head_logits,
            log_return_path=log_return_path,
            temperature=temperature,
        )
        head_weights = torch.softmax(head_logits / temperature, dim=-1)
        if self.portfolio_aggregation == "logit_mean":
            weights = torch.softmax(head_logits.mean(dim=-2) / temperature, dim=-1)
        else:
            weights = head_weights.mean(dim=-2)
        return weights, head_weights

    def refine_policy_logits(
        self,
        head_logits: torch.Tensor,
        log_return_path: Optional[torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        """End-to-end mirror refinement of learned policy logits.

        The Transformer logits remain the optimization state.  At each
        unrolled step their current softmax portfolio is mapped through a
        causal covariance operator, and a signed coefficient learned from the
        portfolio objective updates the logits.  No return label, target
        position, or fixed risk allocation is introduced.
        """

        geometry_enabled = self.lpm_geometry_scale > 0.0
        refinement_enabled = (
            self.policy_refinement_steps > 0
            and self.policy_refinement_scale > 0.0
        )
        if not geometry_enabled and not refinement_enabled:
            return head_logits
        if log_return_path is None:
            raise ValueError(
                "log_return_path is required when policy geometry or refinement is enabled"
            )
        if log_return_path.shape[:-2] != head_logits.shape[:-2] or (
            log_return_path.shape[-2] != self.num_assets
            or log_return_path.shape[-1] != self.lookback_window
        ):
            raise ValueError("log_return_path and policy logits are incompatible")
        all_returns = torch.expm1(log_return_path[..., 1:])
        refined = head_logits
        if geometry_enabled:
            geometry_window = min(
                self.lpm_geometry_window, all_returns.shape[-1]
            )
            geometry_returns = all_returns[..., -geometry_window:]
            downside_scale = torch.mean(
                torch.relu(-geometry_returns).square(), dim=-1
            ).sqrt().add(self.lpm_geometry_epsilon)
            log_geometry = downside_scale.log()
            log_geometry = log_geometry - log_geometry.mean(
                dim=-1, keepdim=True
            )
            geometry_power = self.lpm_geometry_scale * torch.nn.functional.softplus(
                self.lpm_geometry_raw_power
            )
            # This is the closed-form KL-proximal policy update
            #   argmin_w KL(w || softmax(z/T)) + p <w, log(s)>,
            # where s is the causal diagonal LPM metric.  It does not blend a
            # target allocation: every Transformer logit remains in the final
            # softmax and the positive power p is learned end to end.
            refined = refined - (
                temperature
                * geometry_power
                * log_geometry.unsqueeze(-2)
            )
        if not refinement_enabled:
            return refined
        returns = all_returns
        refinement_window = min(
            self.policy_refinement_window, returns.shape[-1]
        )
        returns = returns[..., -refinement_window:]
        covariance = None
        if self.policy_refinement_risk == "variance":
            centered = returns - returns.mean(dim=-1, keepdim=True)
            covariance = torch.einsum(
                "...it,...jt->...ij", centered, centered
            ) / float(centered.shape[-1])
            covariance_scale = covariance.diagonal(
                dim1=-2, dim2=-1
            ).mean(dim=-1, keepdim=True).clamp_min(
                self.signal_normalization_epsilon
            )
            covariance = covariance / covariance_scale.unsqueeze(-1)
        gain = self.policy_refinement_scale * torch.tanh(
            self.policy_refinement_gain
        )
        gain = gain.view(*((1,) * (head_logits.ndim - 2)), -1, 1)
        for _ in range(self.policy_refinement_steps):
            weights = torch.softmax(refined / temperature, dim=-1)
            if covariance is not None:
                marginal_risk = torch.einsum(
                    "...ij,...kj->...ki", covariance, weights
                )
            elif self.policy_refinement_risk == "cvar":
                portfolio_returns = torch.einsum(
                    "...kn,...nt->...kt", weights, returns
                )
                tail_attention = torch.softmax(
                    -portfolio_returns
                    / self.policy_refinement_tail_temperature,
                    dim=-1,
                )
                # Gradient of the smooth historical tail loss with respect
                # to each asset weight.  Tail scenarios depend on the current
                # Transformer portfolio, unlike an asset-wise risk anchor.
                marginal_risk = -torch.einsum(
                    "...kt,...nt->...kn", tail_attention, returns
                )
            else:
                # Gradient of a smooth lower-partial-moment objective.  The
                # downside scenarios are selected by the *current portfolio*
                # return, so this is a genuinely portfolio-level optimization
                # step rather than an asset-wise inverse-risk allocation.  In
                # particular, changing any Transformer logit changes both the
                # portfolio and the scenario weights used by the next step.
                portfolio_returns = torch.einsum(
                    "...kn,...nt->...kt", weights, returns
                )
                scaled_loss = (
                    -portfolio_returns
                    / self.policy_refinement_tail_temperature
                )
                smooth_downside = (
                    self.policy_refinement_tail_temperature
                    * torch.nn.functional.softplus(scaled_loss)
                )
                downside_slope = torch.sigmoid(scaled_loss)
                scenario_weight = smooth_downside * downside_slope
                marginal_risk = -torch.einsum(
                    "...kt,...nt->...kn", scenario_weight, returns
                ) / float(returns.shape[-1])
            marginal_risk = marginal_risk - marginal_risk.mean(
                dim=-1, keepdim=True
            )
            marginal_risk = marginal_risk / marginal_risk.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(self.signal_normalization_epsilon)
            refined = refined - gain * marginal_risk
        return refined

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
        scale_weights = torch.softmax(
            self.risk_gate_logit_scale * self.risk_scale_gate(scale_gate_input),
            dim=-1,
        )
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
            self.risk_gate_logit_scale * self.risk_signal_gate(signal_gate_input),
            dim=-1,
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
        prior_weight = torch.sigmoid(
            self.risk_gate_logit_scale * self.risk_prior_gate(gate_features)
        ).squeeze(-1)
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
            self.risk_gate_logit_scale * self.risk_forecast_gate(forecast_features)
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
        if self.backbone.risk_gate_init == "neutral":
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

    def allocation_head_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        log_return_path: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backbone.allocation_head_logits_from_hidden(
            hidden, log_return_path=log_return_path
        )

    def allocation_weights_from_hidden(
        self,
        hidden: torch.Tensor,
        temperature: float,
        log_return_path: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone.allocation_weights_from_hidden(
            hidden, temperature, log_return_path=log_return_path
        )

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


class DecisionAwarePolicyEnsemble(nn.Module):
    """Jointly trained ensemble of complete decision-aware policy experts.

    Unlike checkpoint ensembling, every expert participates in the same
    end-to-end portfolio objective during training. Their allocation evidence
    is fused inside the model, before the final portfolio is evaluated. This
    averages independent representation and KKT-attention uncertainty without
    injecting any externally specified position.
    """

    def __init__(self, configs, feedback_mode: str = "dual"):
        super().__init__()
        self.policy_experts = int(getattr(configs, "policy_experts", 1))
        if self.policy_experts <= 1:
            raise ValueError("DecisionAwarePolicyEnsemble requires policy_experts > 1")
        self.portfolio_aggregation = str(
            getattr(configs, "portfolio_aggregation", "probability_mean")
        ).lower()
        if self.portfolio_aggregation not in {"probability_mean", "logit_mean"}:
            raise ValueError(
                "portfolio_aggregation must be probability_mean or logit_mean"
            )
        self.experts = nn.ModuleList(
            [
                DecisionAwareModel(configs, feedback_mode=feedback_mode)
                for _ in range(self.policy_experts)
            ]
        )
        self.num_assets = self.experts[0].num_assets
        self.d_model = self.experts[0].d_model

    def initial_forward(
        self, log_return_path: torch.Tensor, date_feats: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        outputs = [
            expert.initial_forward(log_return_path, date_feats)
            for expert in self.experts
        ]
        hidden = tuple(item[0] for item in outputs)
        mu_hat = torch.stack([item[1] for item in outputs], dim=0).mean(dim=0)
        return hidden, mu_hat

    def refine(
        self,
        hidden: Tuple[torch.Tensor, ...],
        kkt_state: Dict[str, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        if not isinstance(hidden, (tuple, list)) or len(hidden) != len(self.experts):
            raise ValueError("hidden must contain one representation per policy expert")
        outputs = [
            expert.refine(expert_hidden, kkt_state)
            for expert, expert_hidden in zip(self.experts, hidden)
        ]
        refined = tuple(item[0] for item in outputs)
        mu_hat = torch.stack([item[1] for item in outputs], dim=0).mean(dim=0)
        return refined, mu_hat

    def normalize_signal(
        self, raw_signal: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return self.experts[0].normalize_signal(raw_signal, sigma)

    def _allocation_head_logits(
        self,
        hidden: Tuple[torch.Tensor, ...],
        log_return_path: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(hidden, (tuple, list)) or len(hidden) != len(self.experts):
            raise ValueError("hidden must contain one representation per policy expert")
        return torch.cat(
            [
                expert.allocation_head_logits_from_hidden(
                    expert_hidden, log_return_path=log_return_path
                )
                for expert, expert_hidden in zip(self.experts, hidden)
            ],
            dim=-2,
        )

    def allocation_logits_from_hidden(
        self, hidden: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self._allocation_head_logits(hidden).mean(dim=-2)

    def allocation_weights_from_hidden(
        self,
        hidden: Tuple[torch.Tensor, ...],
        temperature: float,
        log_return_path: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        head_logits = self._allocation_head_logits(
            hidden, log_return_path=log_return_path
        )
        head_weights = torch.softmax(head_logits / temperature, dim=-1)
        if self.portfolio_aggregation == "logit_mean":
            weights = torch.softmax(head_logits.mean(dim=-2) / temperature, dim=-1)
        else:
            weights = head_weights.mean(dim=-2)
        return weights, head_weights

    def risk_budget_logits(
        self,
        log_return_path: torch.Tensor,
        hidden: Tuple[torch.Tensor, ...],
        kkt_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not isinstance(hidden, (tuple, list)) or len(hidden) != len(self.experts):
            raise ValueError("hidden must contain one representation per policy expert")
        logits = [
            expert.risk_budget_logits(
                log_return_path, expert_hidden, kkt_state=kkt_state
            )
            for expert, expert_hidden in zip(self.experts, hidden)
        ]
        return torch.stack(logits, dim=0).mean(dim=0)
