"""Training and evaluation loop for KKTFormer-v0."""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm.auto import tqdm

from data_provider.data_factory_kkt import data_provider_kkt
from exp.exp_basic import Exp_Basic
from portfolio import (
    DifferentiablePortfolioOptimizer,
    MinimalPortfolioProblem,
    portfolio_cvar_loss,
    sequence_decision_regret_loss,
    compute_kkt_state,
)
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.sit_protocol import SIT_REBALANCE_DATE_SET, SIT_TEST_RANGE


class EXP_KKT(Exp_Basic):
    """KKTFormer-v0 experiment: prediction loss + constrained allocation."""

    def __init__(self, args):
        self.protocol = str(getattr(args, "protocol", "sit")).lower()
        if self.protocol not in {"sit", "native"}:
            raise ValueError("protocol must be one of sit or native")
        self.feedback_mode = str(getattr(args, "feedback_mode", "none")).lower()
        if self.feedback_mode not in {"none", "dual", "jacobian"}:
            raise ValueError("feedback_mode must be one of none, dual, jacobian")
        self.decision_layer = str(
            getattr(args, "decision_layer", "softmax")
        ).lower()
        if self.decision_layer not in {"softmax", "optimizer"}:
            raise ValueError("decision_layer must be one of softmax or optimizer")
        self.temperature = float(getattr(args, "temperature", 1.0))
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        super().__init__(args)
        problem_rebalance_frequency = (
            1 if self.protocol == "sit" else args.rebalance_frequency
        )
        self.problem = MinimalPortfolioProblem(
            num_assets=args.data_pool,
            lookback_window=args.window_size,
            horizon=args.horizon,
            rebalance_frequency=problem_rebalance_frequency,
            eta=args.eta,
            upper_bound=args.upper_bound,
            lower_bound=getattr(args, "lower_bound", 0.0),
            budget_target=getattr(args, "budget_target", 1.0),
        )
        self.portfolio_optimizer = DifferentiablePortfolioOptimizer(
            self.problem,
            num_iterations=args.optimizer_iterations,
            bisection_steps=args.projection_iterations,
            constraint_projection_iterations=getattr(
                args, "constraint_projection_iterations", 20
            ),
            entropy_epsilon=getattr(args, "entropy_epsilon", 1e-4),
        )
        self.probe_optimizer = DifferentiablePortfolioOptimizer(
            self.problem,
            num_iterations=getattr(args, "probe_optimizer_iterations", 5),
            bisection_steps=args.projection_iterations,
            constraint_projection_iterations=getattr(
                args, "constraint_projection_iterations", 20
            ),
            entropy_epsilon=getattr(args, "entropy_epsilon", 1e-4),
        )
        self.entropy_regularization = float(
            getattr(args, "entropy_regularization", 0.0)
        )
        self.entropy_epsilon = float(getattr(args, "entropy_epsilon", 1e-4))
        self.turnover_penalty = float(getattr(args, "turnover_penalty", 0.0))
        self.transaction_cost_smoothing = float(
            getattr(args, "transaction_cost_smoothing", 1e-4)
        )
        self.max_turnover = getattr(args, "max_turnover", None)
        self.gross_exposure_limit = getattr(args, "gross_exposure_limit", None)
        self.sequential_state = bool(getattr(args, "sequential_state", False))
        self._sequential_previous = None
        self.loss_mode = str(getattr(args, "loss_mode", "cvar")).lower()
        if self.loss_mode not in {"cvar", "hybrid"}:
            raise ValueError("sequence KKTFormer supports loss_mode=cvar or hybrid")
        self.regret_weight = float(getattr(args, "regret_weight", 0.1))
        if not math.isfinite(self.regret_weight) or self.regret_weight < 0.0:
            raise ValueError("regret_weight must be finite and non-negative")
        self.prediction_weight = float(
            getattr(args, "prediction_weight", 0.1)
        )

    def _build_model(self):
        model_module = self.model_dict["KKTFormer"]
        if self.feedback_mode == "none":
            model = model_module.Model(self.args).float()
        else:
            model = model_module.DecisionAwareModel(
                self.args,
                feedback_mode=self.feedback_mode,
            ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider_kkt(self.args, flag)

    @staticmethod
    def _parse_bound(value):
        if value is None or value == "":
            return None
        if isinstance(value, (float, int)):
            return float(value)
        pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
        if len(pieces) == 1:
            return float(pieces[0])
        return tuple(float(piece) for piece in pieces)

    def _constraint_inputs(self, batch):
        return {
            "factor_exposure": batch["factor_exposure"].to(
                self.device, non_blocking=True
            ),
            "factor_lower": (
                batch["factor_lower"].to(self.device, non_blocking=True)
                if "factor_lower" in batch
                else self._parse_bound(getattr(self.args, "factor_lower", ""))
            ),
            "factor_upper": (
                batch["factor_upper"].to(self.device, non_blocking=True)
                if "factor_upper" in batch
                else self._parse_bound(getattr(self.args, "factor_upper", ""))
            ),
            "industry_exposure": (
                batch["industry_exposure"].to(self.device, non_blocking=True)
                if "industry_exposure" in batch
                else None
            ),
            "industry_lower": (
                batch["industry_lower"].to(self.device, non_blocking=True)
                if "industry_lower" in batch
                else self._parse_bound(getattr(self.args, "industry_lower", ""))
            ),
            "industry_upper": (
                batch["industry_upper"].to(self.device, non_blocking=True)
                if "industry_upper" in batch
                else self._parse_bound(getattr(self.args, "industry_upper", ""))
            ),
        }

    def _validate_feedback_problem(self, optimizer_kwargs):
        """Keep KKT feedback tied to the problem represented by its KKT state."""

        if self.feedback_mode == "none":
            return
        unsupported = []
        for name in (
            "factor_lower",
            "factor_upper",
            "industry_lower",
            "industry_upper",
            "max_turnover",
            "gross_exposure_limit",
        ):
            if optimizer_kwargs[name] is not None:
                unsupported.append(name)
        if unsupported:
            names = ", ".join(unsupported)
            raise ValueError(
                "KKT feedback currently represents budget, box, quadratic risk, "
                "optional quadratic turnover, and entropy regularization; "
                "unsupported optimizer "
                f"terms: {names}. Use --feedback_mode none for extension-only "
                "experiments."
            )

    def _validate_softmax_problem(self, optimizer_kwargs, lower, upper):
        """Reject hard constraints that a plain simplex softmax cannot enforce."""

        if self.decision_layer != "softmax":
            return
        unsupported = []
        for name in (
            "factor_lower",
            "factor_upper",
            "industry_lower",
            "industry_upper",
            "max_turnover",
            "gross_exposure_limit",
        ):
            if optimizer_kwargs[name] is not None:
                unsupported.append(name)
        if not torch.allclose(lower, torch.zeros_like(lower), atol=1e-7):
            unsupported.append("lower_bounds")
        if not torch.allclose(upper, torch.ones_like(upper), atol=1e-7):
            unsupported.append("upper_bounds")
        if abs(float(self.problem.budget_target) - 1.0) > 1e-7:
            unsupported.append("budget_target")
        if unsupported:
            raise ValueError(
                "decision_layer=softmax only enforces the long-only unit simplex; "
                f"unsupported hard constraints: {', '.join(unsupported)}. "
                "Use --decision_layer optimizer for constrained experiments."
            )

    def _softmax_state(self, weights, lower, upper, w_prev):
        """Return the diagnostics expected by training and test exporters."""

        epsilon = self.entropy_epsilon
        entropy = (-weights * torch.log(weights.clamp_min(epsilon))).sum(dim=-1)
        smoothed_entropy_term = (
            (weights + epsilon) * torch.log(weights + epsilon)
            - epsilon * math.log(epsilon)
        ).sum(dim=-1)
        active = torch.cat(
            (
                (weights - lower <= 1e-5).to(weights.dtype),
                (upper - weights <= 1e-5).to(weights.dtype),
            ),
            dim=-1,
        )
        return {
            "budget_residual": weights.sum(dim=-1) - self.problem.budget_target,
            "lower_violation": torch.relu(lower - weights),
            "upper_violation": torch.relu(weights - upper),
            "net_exposure": weights.sum(dim=-1),
            "gross_exposure": weights.abs().sum(dim=-1),
            "active_constraint_ratio": active.mean(dim=-1),
            "entropy": entropy,
            "entropy_penalty": self.entropy_regularization * smoothed_entropy_term,
            "turnover": (
                (weights - w_prev).abs().sum(dim=-1)
                if w_prev is not None
                else torch.zeros_like(entropy)
            ),
        }

    def _forward_batch(self, batch, w_prev_override=None):
        log_return_path = batch["log_return_path"].to(self.device, non_blocking=True)
        date_feats = batch["date_feats"].to(self.device, non_blocking=True)
        future_returns = batch["future_returns"].to(self.device, non_blocking=True)
        sigma = batch["Sigma"].to(self.device, non_blocking=True)
        w_prev = (
            w_prev_override.to(self.device, non_blocking=True)
            if w_prev_override is not None
            else batch["w_prev"].to(self.device, non_blocking=True)
        )
        transaction_cost_rate = batch["transaction_cost_rate"].to(
            self.device, non_blocking=True
        )

        constraint_inputs = self._constraint_inputs(batch)
        lower_bounds = batch["lower_bounds"].to(self.device, non_blocking=True)
        upper_bounds = batch["upper_bounds"].to(self.device, non_blocking=True)

        batch_size, horizon, num_assets = future_returns.shape

        def repeat_horizon(value):
            if value is None or not isinstance(value, torch.Tensor):
                return value
            if value.shape[0] != batch_size:
                return value
            expanded = value.unsqueeze(1).expand(
                batch_size, horizon, *value.shape[1:]
            )
            return expanded.reshape(batch_size * horizon, *value.shape[1:])

        if sigma.shape != (batch_size, horizon, num_assets, num_assets):
            raise ValueError(
                "Sigma must have shape (B, H, N, N); rebuild legacy context "
                "caches instead of broadcasting Sigma_t across the horizon"
            )
        factor_sequence = constraint_inputs["factor_exposure"]
        if (
            factor_sequence.ndim != 4
            or factor_sequence.shape[:3] != (batch_size, horizon, num_assets)
        ):
            raise ValueError(
                "factor_exposure must have shape (B, H, N, K); rebuild legacy "
                "context caches instead of broadcasting F_t across the horizon"
            )

        sigma_flat = sigma.reshape(batch_size * horizon, num_assets, num_assets)
        w_prev_flat = repeat_horizon(w_prev)
        lower_flat = repeat_horizon(lower_bounds)
        upper_flat = repeat_horizon(upper_bounds)
        factor_flat = factor_sequence.reshape(
            batch_size * horizon, num_assets, factor_sequence.shape[-1]
        )
        industry_flat = repeat_horizon(constraint_inputs["industry_exposure"])
        factor_lower_flat = repeat_horizon(constraint_inputs["factor_lower"])
        factor_upper_flat = repeat_horizon(constraint_inputs["factor_upper"])
        industry_lower_flat = repeat_horizon(constraint_inputs["industry_lower"])
        industry_upper_flat = repeat_horizon(constraint_inputs["industry_upper"])
        cost_rate_flat = repeat_horizon(transaction_cost_rate).reshape(-1)

        optimizer_kwargs = {
            "factor_exposure": factor_flat,
            "factor_lower": factor_lower_flat,
            "factor_upper": factor_upper_flat,
            "industry_exposure": industry_flat,
            "industry_lower": industry_lower_flat,
            "industry_upper": industry_upper_flat,
            "w_prev": w_prev_flat,
            "lower_bounds": lower_flat,
            "upper_bounds": upper_flat,
            "turnover_penalty": self.turnover_penalty,
            # In the strict KKT method, trading cost belongs to the realized
            # decision loss/evaluation, not the probe or final optimizer. The
            # extension-only no-feedback optimizer retains its smooth cost.
            "transaction_cost_rate": (
                cost_rate_flat if self.feedback_mode == "none" else None
            ),
            "transaction_cost_smoothing": self.transaction_cost_smoothing,
            "max_turnover": self.max_turnover,
            "gross_exposure_limit": self.gross_exposure_limit,
            "entropy_regularization": self.entropy_regularization,
        }
        self._validate_feedback_problem(optimizer_kwargs)
        self._validate_softmax_problem(optimizer_kwargs, lower_flat, upper_flat)

        def reshape_state(flat_state):
            result = {}
            for key, value in flat_state.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size * horizon:
                    result[key] = value.reshape(batch_size, horizon, *value.shape[1:])
                else:
                    result[key] = value
            return result

        model_core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if self.feedback_mode == "none":
            allocation_hidden, raw_mu_hat = self.model(log_return_path, date_feats)
            mu_hat = model_core.normalize_signal(raw_mu_hat, sigma)
            initial_weights = None
        else:
            hidden0, raw_mu0 = model_core.initial_forward(log_return_path, date_feats)
            mu0 = model_core.normalize_signal(raw_mu0, sigma)
            mu0_flat = mu0.reshape(batch_size * horizon, num_assets)
            probe_weights, _ = self.probe_optimizer(
                mu_hat=mu0_flat,
                sigma=sigma_flat,
                **optimizer_kwargs,
            )
            kkt_state_flat = compute_kkt_state(
                mu_hat=mu0_flat,
                sigma=sigma_flat,
                weights=probe_weights,
                upper_bounds=upper_flat,
                problem=self.problem,
                lower_bounds=lower_flat,
                active_tolerance=float(
                    getattr(self.args, "active_tolerance", 1e-5)
                ),
                compute_jacobian=self.feedback_mode == "jacobian",
                w_prev=w_prev_flat,
                turnover_penalty=self.turnover_penalty,
                entropy_regularization=self.entropy_regularization,
                entropy_epsilon=self.entropy_epsilon,
            )
            kkt_state = reshape_state(kkt_state_flat)
            allocation_hidden, raw_mu_hat = model_core.refine(hidden0, kkt_state)
            mu_hat = model_core.normalize_signal(raw_mu_hat, sigma)
            initial_weights = probe_weights

        if self.decision_layer == "softmax":
            allocation_logits = model_core.allocation_logits_from_hidden(
                allocation_hidden
            )
            weights = torch.softmax(allocation_logits / self.temperature, dim=-1)
            weights_flat = weights.reshape(batch_size * horizon, num_assets)
            state = reshape_state(
                self._softmax_state(
                    weights_flat, lower_flat, upper_flat, w_prev_flat
                )
            )
        else:
            weights_flat, state_flat = self.portfolio_optimizer(
                mu_hat=mu_hat.reshape(batch_size * horizon, num_assets),
                sigma=sigma_flat,
                initial_weights=initial_weights,
                **optimizer_kwargs,
            )
            weights = weights_flat.reshape(batch_size, horizon, num_assets)
            state = reshape_state(state_flat)
        if self.feedback_mode != "none":
            # This diagnostic belongs to the short-run probe optimizer, not
            # the final softmax policy (or optional optimizer ablation).
            state["probe_kkt_stationarity_residual_inf"] = kkt_state[
                "kkt_stationarity_residual_inf"
            ]
            state["probe_entropy_gradient"] = kkt_state["entropy_gradient"]
            state["probe_entropy_curvature"] = kkt_state["entropy_curvature"]

        cvar_batch, cvar_components = portfolio_cvar_loss(
            weights=weights,
            future_returns=future_returns,
            alpha=float(getattr(self.args, "cvar_alpha", 0.95)),
            variant=str(getattr(self.args, "cvar_variant", "sit")),
            smooth_temperature=float(
                getattr(self.args, "cvar_temperature", 1e-3)
            ),
            w_prev=w_prev,
            # The strict SIT protocol uses CVaR alone as its training
            # objective. Costs and turnover are evaluated separately.
            transaction_cost_rate=(
                None if self.protocol == "sit" else transaction_cost_rate
            ),
            turnover_penalty=(
                0.0 if self.protocol == "sit" else self.turnover_penalty
            ),
            transaction_cost_smoothing=self.transaction_cost_smoothing,
        )
        cvar_loss = cvar_batch.mean()
        regret_loss = cvar_loss.detach() * 0.0
        oracle_objective = cvar_loss.detach() * 0.0
        mean_return_loss = cvar_loss.detach() * 0.0
        oracle_return_loss = cvar_loss.detach() * 0.0

        if self.loss_mode == "hybrid":
            # The hindsight oracle sees the realized next-period return for
            # each causal token. It starts from the detached model decision,
            # uses the exact same feasible set, and never receives gradients.
            with torch.no_grad():
                oracle_weights_flat, _ = self.portfolio_optimizer(
                    mu_hat=future_returns.reshape(
                        batch_size * horizon, num_assets
                    ),
                    sigma=sigma_flat,
                    initial_weights=weights_flat.detach(),
                    **optimizer_kwargs,
                )
            oracle_weights = oracle_weights_flat.reshape(
                batch_size, horizon, num_assets
            )
            regret_batch, regret_components = sequence_decision_regret_loss(
                predicted_weights=weights,
                oracle_weights=oracle_weights,
                realized_returns=future_returns,
                sigma=sigma,
                problem=self.problem,
                w_prev=w_prev_flat.reshape(batch_size, horizon, num_assets),
                transaction_cost_rate=(
                    optimizer_kwargs["transaction_cost_rate"].reshape(
                        batch_size, horizon
                    )
                    if optimizer_kwargs["transaction_cost_rate"] is not None
                    else None
                ),
                turnover_penalty=self.turnover_penalty,
                transaction_cost_smoothing=self.transaction_cost_smoothing,
                entropy_regularization=self.entropy_regularization,
                entropy_epsilon=self.entropy_epsilon,
            )
            regret_loss = regret_batch.mean()
            oracle_objective = regret_components["oracle_objective"].mean()
            mean_return_loss = regret_components["predicted_return_loss"].mean()
            oracle_return_loss = regret_components["oracle_return_loss"].mean()

        total_loss = cvar_loss + self.regret_weight * regret_loss
        zero = total_loss.detach() * 0.0

        details = {
            "total_loss": total_loss,
            "prediction_loss": zero,
            "utility_loss": zero,
            "cvar_loss": cvar_loss,
            "regret_loss": regret_loss,
            "weighted_regret_loss": self.regret_weight * regret_loss,
            "mean_return_loss": mean_return_loss,
            "oracle_return_loss": oracle_return_loss,
            "risk_loss": zero,
            "entropy": state["entropy"].mean(),
            "entropy_penalty": state["entropy_penalty"].mean(),
            "transaction_cost": cvar_components["transaction_cost"].mean(),
            "cvar_var": cvar_components["var"].mean(),
            "oracle_objective": oracle_objective,
            "feedback_mode": self.feedback_mode,
            "decision_layer": self.decision_layer,
            "turnover": cvar_components["turnover"].mean(),
            "smooth_transaction_cost": cvar_components["smooth_transaction_cost"].mean(),
        }
        for key in (
            "active_constraint_ratio",
            "turnover_violation",
            "gross_exposure_violation",
            "factor_lower_violation",
            "factor_upper_violation",
            "industry_lower_violation",
            "industry_upper_violation",
        ):
            if key in state:
                value = state[key].detach()
                details[key] = (
                    value.mean()
                    if value.numel() > 0
                    else torch.zeros((), dtype=value.dtype, device=value.device)
                )
        return total_loss, mu_hat, weights, state, future_returns, details

    def _validation_loss(self, loader, epoch: int) -> float:
        self.model.eval()
        losses = []
        self._sequential_previous = None
        with torch.no_grad():
            progress = tqdm(
                loader,
                desc=f"Validate {epoch + 1}/{self.args.train_epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            for batch in progress:
                previous = self._sequential_previous if self.sequential_state else None
                _, _, weights, _, _, details = self._forward_batch(
                    batch, w_prev_override=previous
                )
                losses.append(float(details["total_loss"].item()))
                if self.sequential_state:
                    self._sequential_previous = weights[:, 0].detach()
        if not losses:
            raise RuntimeError("validation loader is empty")
        return float(np.mean(losses))

    def train(self, setting):
        train_data, train_loader = self._get_data("train")
        _, val_loader = self._get_data("val")
        del train_data

        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
        )
        model_optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            self._sequential_previous = None
            train_losses = []
            prediction_losses = []
            utility_losses = []
            cvar_losses = []
            regret_losses = []
            progress = tqdm(
                train_loader,
                desc=f"Train {epoch + 1}/{self.args.train_epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in progress:
                model_optimizer.zero_grad(set_to_none=True)
                previous = self._sequential_previous if self.sequential_state else None
                loss, _, weights, _, _, details = self._forward_batch(
                    batch, w_prev_override=previous
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                model_optimizer.step()
                if self.sequential_state:
                    self._sequential_previous = weights[:, 0].detach()
                train_losses.append(float(loss.item()))
                prediction_losses.append(float(details["prediction_loss"].item()))
                utility_losses.append(float(details["utility_loss"].item()))
                cvar_losses.append(float(details["cvar_loss"].item()))
                regret_losses.append(float(details["regret_loss"].item()))
                progress.set_postfix(loss=f"{train_losses[-1]:.6f}")

            val_loss = self._validation_loss(val_loader, epoch)
            train_loss = float(np.mean(train_losses)) if train_losses else math.nan
            print(
                f"[Epoch {epoch + 1}] mode={self.loss_mode} "
                f"train_loss={train_loss:.8f} "
                f"train_prediction={np.mean(prediction_losses):.8f} "
                f"train_utility={np.mean(utility_losses):.8f} "
                f"train_cvar={np.mean(cvar_losses):.8f} "
                f"train_regret={np.mean(regret_losses):.8f} "
                f"val_loss={val_loss:.8f}"
            )
            early_stopping(val_loss, self.model, str(checkpoint_dir))
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            adjust_learning_rate(model_optimizer, epoch + 1, self.args)

        checkpoint = checkpoint_dir / "checkpoint.pth"
        if checkpoint.exists():
            self.model.load_state_dict(
                torch.load(checkpoint, map_location=self.device, weights_only=True)
            )
        return self.model

    @staticmethod
    def _decode_date_batch(value, batch_size: int) -> List[List[str]]:
        """Convert default-collated future dates from H x B into B x H."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(value, str):
            if batch_size != 1:
                raise ValueError("one date string cannot describe multiple samples")
            return [[value]]
        if not isinstance(value, (tuple, list)) or not value:
            return [[] for _ in range(batch_size)]

        # Dataset items contain a list of H strings. PyTorch's default collate
        # transposes these into H sequences, each containing B strings.
        if all(isinstance(item, (tuple, list)) for item in value):
            if any(len(item) != batch_size for item in value):
                raise ValueError("collated future_dates has an invalid batch dimension")
            return [
                [str(horizon_dates[sample_index]) for horizon_dates in value]
                for sample_index in range(batch_size)
            ]
        if batch_size == 1 and all(isinstance(item, str) for item in value):
            return [list(value)]
        raise ValueError("unsupported future_dates collation structure")

    @staticmethod
    def _portfolio_metrics(daily_returns: pd.Series) -> Dict[str, float]:
        daily_returns = daily_returns.dropna().sort_index()
        if daily_returns.empty:
            raise RuntimeError("no daily portfolio returns were generated")
        returns = daily_returns.to_numpy(dtype=float)
        equity = np.cumprod(1.0 + returns)
        annual_factor = 252.0
        volatility_daily = float(np.std(returns))
        sharpe = float(
            returns.mean() / (volatility_daily + 1e-12) * math.sqrt(annual_factor)
        )
        negative_returns = returns[returns < 0]
        negative_std = float(np.std(negative_returns)) if negative_returns.size else math.nan
        sortino = float(
            returns.mean() / (negative_std + 1e-12) * math.sqrt(annual_factor)
        )
        peak = np.maximum.accumulate(equity)
        max_drawdown = float(np.max((peak - equity) / (peak + 1e-12)))
        final_wealth_factor = float(equity[-1])
        annual_return = float(
            final_wealth_factor ** (annual_factor / len(returns)) - 1.0
        )
        return {
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDrawdown": max_drawdown,
            "AnnualReturn": annual_return,
            "AnnualVol": float(volatility_daily * math.sqrt(annual_factor)),
            "FinalWealthFactor": final_wealth_factor,
            "WinRate": float((returns > 0).mean()),
        }

    def _hold_last_position_returns(
        self,
        start_date: str,
        end_date: str,
        weights: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Extend the final complete horizon by holding its last portfolio.

        The 20-observation test decision grid can end before the final date of
        the test split.  The final decision remains investable after its
        cached horizon, so the backtest must carry that position forward to
        the common evaluation endpoint instead of silently dropping the tail.
        """

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if end <= start:
            return []

        csv_path = Path(self.args.root_path) / self.args.data_path
        prices = (
            pd.read_csv(csv_path, parse_dates=["Date"])
            .set_index("Date")
            .iloc[:, : self.args.data_pool]
            .sort_index()
        )
        start_position = int(prices.index.searchsorted(start, side="left"))
        end_position = int(prices.index.searchsorted(end, side="right"))
        if start_position >= len(prices) or end_position <= start_position:
            return []

        # Include the last known price at start_date so the first tail return
        # is measured from the end of the cached horizon.
        begin = max(0, start_position - 1)
        window = prices.iloc[begin:end_position]
        returns = window.pct_change().iloc[1:]
        returns = returns.loc[(returns.index > start) & (returns.index <= end)]
        return [
            (date.strftime("%Y-%m-%d"), float(row.to_numpy().dot(weights)))
            for date, row in returns.iterrows()
        ]

    def eval(self, setting, load=True):
        """Evaluate with the released SIT record/dedup protocol.

        SIT iterates every horizon token from each overlapping test window and
        keeps the first representation encountered for each date. Thus most
        newly encountered dates use a late-horizon causal representation.
        """

        if load:
            checkpoint = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            if checkpoint.exists():
                self.model.load_state_dict(
                    torch.load(checkpoint, map_location=self.device, weights_only=True)
                )
        self.model.eval()
        _, test_loader = self._get_data("test")

        event_records = {}
        prediction_losses = []
        total_losses = []
        utility_losses = []
        cvar_losses = []
        regret_losses = []
        event_turnovers = []
        event_transaction_costs = []
        budget_errors = []
        lower_violations = []
        upper_violations = []
        factor_violations = []
        industry_violations = []
        turnover_violations = []
        gross_violations = []
        active_constraint_ratios = []
        net_exposures = []
        gross_exposures = []
        factor_exposure_values = []
        industry_exposure_values = []
        probe_kkt_residual_infs = []
        asset_names = [f"Asset_{i}" for i in range(self.args.data_pool)]
        held_weights = torch.zeros(
            1, self.args.data_pool, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            progress = tqdm(
                test_loader,
                desc="Test",
                unit="context",
                dynamic_ncols=True,
            )
            for batch in progress:
                batch_size = int(batch["future_returns"].shape[0])
                execution_date_batch = self._decode_date_batch(
                    batch["future_dates"], batch_size
                )
                if any(not dates for dates in execution_date_batch):
                    raise RuntimeError("test context has no execution date")
                _, _, weights, state, _, details = self._forward_batch(
                    batch,
                    # A generated previous position is inherently sequential.
                    # Batched evaluation uses each context's cached w_prev;
                    # --sequential_state keeps the loader at batch size one.
                    w_prev_override=held_weights if batch_size == 1 else None,
                )
                # Each detail is a batch mean. Repeat it by the batch size so
                # the final diagnostic mean does not overweight a short tail.
                total_losses.extend([float(details["total_loss"].item())] * batch_size)
                prediction_losses.extend(
                    [float(details["prediction_loss"].item())] * batch_size
                )
                utility_losses.extend(
                    [float(details["utility_loss"].item())] * batch_size
                )
                cvar_losses.extend([float(details["cvar_loss"].item())] * batch_size)
                regret_losses.extend(
                    [float(details["regret_loss"].item())] * batch_size
                )

                # Preserve the exact batch-size-one ordering: samples first,
                # then horizon tokens. This keeps SIT's first-date dedup rule.
                for sample_index, execution_dates in enumerate(execution_date_batch):
                    token_indices = (
                        range(len(execution_dates)) if self.protocol == "sit" else (0,)
                    )
                    for token_index in token_indices:
                        execution_date = execution_dates[token_index]
                        if execution_date in event_records:
                            continue
                        token_weights = weights[sample_index, token_index].detach()
                        is_rebalance = (
                            not event_records
                            or self.protocol != "sit"
                            or execution_date in SIT_REBALANCE_DATE_SET
                        )
                        if is_rebalance and batch_size == 1:
                            held_weights = token_weights.unsqueeze(0)
                        event_records[execution_date] = {
                            "weight": token_weights.cpu().numpy(),
                            "state": state,
                            "horizon_token": token_index,
                        }

                budget_errors.extend(
                    state["budget_residual"].detach().cpu().numpy().reshape(-1).tolist()
                )
                lower_violations.append(
                    float(state["lower_violation"].detach().cpu().numpy().max())
                )
                upper_violations.append(
                    float(state["upper_violation"].detach().cpu().numpy().max())
                )
                net_exposures.extend(
                    state["net_exposure"].detach().cpu().numpy().reshape(-1).tolist()
                )
                gross_exposures.extend(
                    state["gross_exposure"].detach().cpu().numpy().reshape(-1).tolist()
                )
                active_constraint_ratios.extend(
                    state["active_constraint_ratio"].detach().cpu().numpy().reshape(-1).tolist()
                )
                if "probe_kkt_stationarity_residual_inf" in state:
                    probe_kkt_residual_infs.extend(
                        state["probe_kkt_stationarity_residual_inf"]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .tolist()
                    )
                if "turnover_violation" in state:
                    turnover_violations.extend(
                        state["turnover_violation"].detach().cpu().numpy().reshape(-1).tolist()
                    )
                if "gross_exposure_violation" in state:
                    gross_violations.extend(
                        state["gross_exposure_violation"].detach().cpu().numpy().reshape(-1).tolist()
                    )
                for key in ("factor_lower_violation", "factor_upper_violation"):
                    if key in state:
                        violation = state[key].detach()
                        # The optimizer represents an unconstrained factor
                        # block as a valid tensor with shape ``(..., 0)``.
                        # Such a block has no maximum; it means that this
                        # constraint family is disabled, not that eval failed.
                        if violation.numel() > 0:
                            factor_violations.append(float(violation.max().item()))
                if "factor_exposure" in state:
                    factor_exposure_values.append(
                        float(np.abs(state["factor_exposure"].detach().cpu().numpy()).max())
                    )
                for key in ("industry_lower_violation", "industry_upper_violation"):
                    if key in state:
                        violation = state[key].detach()
                        if violation.numel() > 0:
                            industry_violations.append(float(violation.max().item()))
                if "industry_exposure" in state:
                    industry_exposure_values.append(
                        float(np.abs(state["industry_exposure"].detach().cpu().numpy()).max())
                    )

        if self.protocol == "sit":
            evaluation_start, evaluation_end = SIT_TEST_RANGE
        else:
            evaluation_start = "2020-01-01"
            evaluation_end = getattr(self.args, "evaluation_end_date", "2024-12-31")

        csv_path = Path(self.args.root_path) / self.args.data_path
        prices = (
            pd.read_csv(csv_path, parse_dates=["Date"])
            .set_index("Date")
            .iloc[:, : self.args.data_pool]
            .sort_index()
        )
        start = pd.Timestamp(evaluation_start)
        end = pd.Timestamp(evaluation_end)
        start_position = int(prices.index.searchsorted(start, side="left"))
        end_position = int(prices.index.searchsorted(end, side="right"))
        if end_position <= start_position:
            raise RuntimeError("invalid SIT evaluation price range")
        # SIT labels the return P[t+1] / P[t] - 1 with date t.  A backward
        # pct_change would shift every realized return by one day.
        forward_returns = prices.shift(-1).divide(prices) - 1.0
        daily_returns = forward_returns.loc[
            (forward_returns.index >= start) & (forward_returns.index <= end)
        ].dropna()
        if self.protocol == "sit":
            # The released SIT windows stop at the final date that still has a
            # next-day price and belongs to the union of their H-step outputs.
            available_prediction_dates = sorted(event_records)
            if not available_prediction_dates:
                raise RuntimeError("no KKT test predictions were generated")
            last_prediction_date = pd.Timestamp(available_prediction_dates[-1])
            daily_returns = daily_returns.loc[:last_prediction_date]

        transaction_cost_bps = float(
            getattr(self.args, "trade_cost_bps", 0.0)
            if getattr(self.args, "transaction_cost_bps", None) is None
            else self.args.transaction_cost_bps
        )
        cost_rate = transaction_cost_bps * 1e-4
        initial_capital = 10_000.0
        capital = initial_capital
        current_w = None
        previous_w = np.zeros(self.args.data_pool, dtype=float)
        daily_return_rows: List[Tuple[str, float]] = []
        position_rows = []
        equity_curve = [capital]

        for date, asset_return in daily_returns.iterrows():
            date_string = date.strftime("%Y-%m-%d")
            cost_cash = 0.0
            turnover = 0.0
            should_rebalance = current_w is None or (
                self.protocol == "sit" and date_string in SIT_REBALANCE_DATE_SET
            ) or (
                self.protocol != "sit" and date_string in event_records
            )
            if should_rebalance:
                if date_string not in event_records:
                    raise RuntimeError(
                        f"missing KKT decision for SIT execution date {date_string}"
                    )
                current_w = event_records[date_string]["weight"].astype(float)
                turnover = float(np.abs(current_w - previous_w).sum())
                cost_cash = turnover * cost_rate * capital
                previous_w = current_w.copy()

            previous_capital = capital
            pnl = float(np.dot(current_w, asset_return.to_numpy())) * previous_capital
            pnl -= cost_cash
            capital += pnl
            daily_return = pnl / (previous_capital + 1e-12)
            daily_return_rows.append((date_string, daily_return))
            equity_curve.append(capital)
            position_row = {
                "Date": date_string,
                "DailyPnL": pnl,
                "Cost": cost_cash,
                "Turnover": turnover if cost_cash > 0.0 else 0.0,
                "CumulativeCapital": capital,
            }
            position_row.update(
                {
                    f"Weight_{name}": float(value)
                    for name, value in zip(asset_names, current_w)
                }
            )
            position_rows.append(position_row)
            if turnover > 0.0 or cost_cash > 0.0:
                event_turnovers.append(turnover)
                event_transaction_costs.append(cost_rate * turnover)

        result_dir = Path(self.args.results_path) / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        positions = pd.DataFrame(position_rows)
        positions.to_csv(result_dir / "test_positions.csv", index=False)

        daily = pd.DataFrame(daily_return_rows, columns=["Date", "Return"])
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = (
            daily.groupby("Date", as_index=False)["Return"]
            .last()
            .sort_values("Date")
        )
        daily_series = pd.Series(daily["Return"].to_numpy(), index=daily["Date"])
        daily["Equity"] = (1.0 + daily["Return"]).cumprod()
        daily.to_csv(result_dir / "test_daily_returns.csv", index=False)

        # Keep test_metrics.csv exactly compatible with SIT's seven reported
        # metrics.  KKT-specific diagnostics are written separately so they
        # do not change the baseline comparison table.
        metrics = self._portfolio_metrics(daily_series)
        diagnostics = {
            "Protocol": self.protocol,
            "LossMode": self.loss_mode,
            "DecisionLayer": self.decision_layer,
            "SoftmaxTemperature": self.temperature,
            "RegretWeight": self.regret_weight,
            "CVaRVariant": str(getattr(self.args, "cvar_variant", "sit")),
            "EntropyRegularization": self.entropy_regularization,
            "EntropyEpsilon": self.entropy_epsilon,
            "TestLoss": float(np.mean(total_losses)) if total_losses else math.nan,
            "PredictionLoss": float(np.mean(prediction_losses))
            if prediction_losses else math.nan,
            "UtilityLoss": float(np.mean(utility_losses))
            if utility_losses else math.nan,
            "CVaRLoss": float(np.mean(cvar_losses)) if cvar_losses else math.nan,
            "DecisionRegret": float(np.mean(regret_losses))
            if regret_losses else math.nan,
            "MeanTurnover": float(np.mean(event_turnovers))
            if event_turnovers else 0.0,
            "MaxAbsBudgetResidual": float(np.max(np.abs(budget_errors))),
            "MaxLowerViolation": float(np.max(lower_violations)),
            "MaxUpperViolation": float(np.max(upper_violations)),
            "MeanTransactionCost": float(np.mean(event_transaction_costs))
            if event_transaction_costs else 0.0,
            "MeanNetExposure": float(np.mean(net_exposures)),
            "MeanGrossExposure": float(np.mean(gross_exposures)),
            "MeanActiveConstraintRatio": float(np.mean(active_constraint_ratios)),
            "MeanProbeKKTStationarityResidualInf": (
                float(np.mean(probe_kkt_residual_infs))
                if probe_kkt_residual_infs
                else math.nan
            ),
            "MaxProbeKKTStationarityResidualInf": (
                float(np.max(probe_kkt_residual_infs))
                if probe_kkt_residual_infs
                else math.nan
            ),
            "EvaluationEndDate": str(evaluation_end),
            "MaxTurnoverViolation": float(np.max(turnover_violations))
            if turnover_violations else 0.0,
            "MaxGrossExposureViolation": float(np.max(gross_violations))
            if gross_violations else 0.0,
            "MaxFactorExposureViolation": float(np.max(factor_violations))
            if factor_violations else 0.0,
            "MaxIndustryExposureViolation": float(np.max(industry_violations))
            if industry_violations else 0.0,
            "MaxFactorExposure": float(np.max(factor_exposure_values))
            if factor_exposure_values else 0.0,
            "MaxIndustryExposure": float(np.max(industry_exposure_values))
            if industry_exposure_values else 0.0,
            "SequentialState": True,
            "NumTestContexts": float(len(event_records)),
            "NumCompleteHorizonContexts": float(len(total_losses)),
            "NumDailyObservations": float(len(daily_series)),
            "TotalReturn": float(metrics["FinalWealthFactor"] - 1.0),
        }
        pd.DataFrame([metrics]).to_csv(result_dir / "test_metrics.csv", index=False)
        pd.DataFrame([diagnostics]).to_csv(
            result_dir / "test_diagnostics.csv", index=False
        )

        plt.figure(figsize=(10, 4))
        plt.plot(daily["Date"], daily["Equity"], label="KKTFormer-v0")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / "test_equity_curve.png", dpi=150)
        plt.close()
        print(f"[Test] results saved to {result_dir}")
        print({**metrics, **diagnostics})
        return metrics
