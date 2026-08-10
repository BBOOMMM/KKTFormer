"""Training and evaluation loop for KKTFormer-v0."""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim

from data_provider.data_factory_kkt import data_provider_kkt
from exp.exp_basic import Exp_Basic
from portfolio import (
    DifferentiablePortfolioOptimizer,
    MinimalPortfolioProblem,
    decision_regret_loss,
    portfolio_cvar_loss,
    portfolio_objective_loss,
)
from utils.tools import EarlyStopping, adjust_learning_rate


class EXP_KKT(Exp_Basic):
    """KKTFormer-v0 experiment: prediction loss + constrained allocation."""

    def __init__(self, args):
        super().__init__(args)
        self.problem = MinimalPortfolioProblem(
            num_assets=args.data_pool,
            lookback_window=args.window_size,
            horizon=args.horizon,
            rebalance_frequency=args.rebalance_frequency,
            eta=args.eta,
            upper_bound=args.upper_bound,
        )
        self.portfolio_optimizer = DifferentiablePortfolioOptimizer(
            self.problem,
            num_iterations=args.optimizer_iterations,
            bisection_steps=args.projection_iterations,
        )
        self.loss_mode = str(getattr(args, "loss_mode", "prediction")).lower()
        if self.loss_mode not in {
            "prediction",
            "utility",
            "cvar",
            "regret",
            "hybrid",
        }:
            raise ValueError(
                "loss_mode must be one of prediction, utility, cvar, regret, hybrid"
            )
        self.prediction_weight = float(
            getattr(args, "prediction_weight", 0.1)
        )

    def _build_model(self):
        model = self.model_dict["KKTFormer"].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider_kkt(self.args, flag)

    def _forward_batch(self, batch):
        market_window = batch["market_window"].to(self.device, non_blocking=True)
        future_returns = batch["future_returns"].to(self.device, non_blocking=True)
        sigma = batch["Sigma"].to(self.device, non_blocking=True)
        upper_bounds = batch["upper_bounds"].to(self.device, non_blocking=True)
        w_prev = batch["w_prev"].to(self.device, non_blocking=True)
        transaction_cost_rate = batch["transaction_cost_rate"].to(
            self.device, non_blocking=True
        )

        _, mu_hat = self.model(market_window)
        weights, state = self.portfolio_optimizer(
            mu_hat=mu_hat,
            sigma=sigma,
            upper_bounds=upper_bounds,
        )
        target_mu = self.problem.aggregate_future_returns(future_returns)
        if str(getattr(self.args, "prediction_loss", "MSE")).upper() == "HUBER":
            prediction_loss = F.smooth_l1_loss(mu_hat, target_mu)
        else:
            prediction_loss = F.mse_loss(mu_hat, target_mu)

        utility_batch, utility_components = portfolio_objective_loss(
            weights=weights,
            future_returns=future_returns,
            sigma=sigma,
            problem=self.problem,
            w_prev=w_prev,
            transaction_cost_rate=transaction_cost_rate,
        )
        utility_loss = utility_batch.mean()

        cvar_batch, cvar_components = portfolio_cvar_loss(
            weights=weights,
            future_returns=future_returns,
            alpha=float(getattr(self.args, "cvar_alpha", 0.95)),
            smooth_temperature=float(
                getattr(self.args, "cvar_temperature", 1e-3)
            ),
            w_prev=w_prev,
            transaction_cost_rate=transaction_cost_rate,
        )
        cvar_loss = cvar_batch.mean()

        if self.loss_mode in {"regret", "hybrid"}:
            # The oracle uses the same optimizer, risk matrix and constraints;
            # future returns enter only through the oracle target and loss.
            with torch.no_grad():
                oracle_weights, _ = self.portfolio_optimizer(
                    mu_hat=target_mu,
                    sigma=sigma,
                    upper_bounds=upper_bounds,
                )
            regret_batch, regret_components = decision_regret_loss(
                predicted_weights=weights,
                oracle_weights=oracle_weights,
                future_returns=future_returns,
                sigma=sigma,
                problem=self.problem,
                w_prev=w_prev,
                transaction_cost_rate=transaction_cost_rate,
            )
            regret_loss = regret_batch.mean()
        else:
            regret_loss = prediction_loss.detach() * 0.0
            regret_components = {
                "predicted_objective": utility_batch.detach(),
                "oracle_objective": utility_batch.detach(),
                "regret": utility_batch.detach() * 0.0,
            }

        if self.loss_mode == "prediction":
            total_loss = prediction_loss
        elif self.loss_mode == "utility":
            total_loss = utility_loss
        elif self.loss_mode == "cvar":
            total_loss = cvar_loss
        elif self.loss_mode == "regret":
            total_loss = regret_loss
        else:
            total_loss = regret_loss + self.prediction_weight * prediction_loss

        details = {
            "total_loss": total_loss,
            "prediction_loss": prediction_loss,
            "utility_loss": utility_loss,
            "cvar_loss": cvar_loss,
            "regret_loss": regret_loss,
            "mean_return_loss": utility_components["return_loss"].mean(),
            "risk_loss": utility_components["risk_loss"].mean(),
            "transaction_cost": utility_components["transaction_cost"].mean(),
            "cvar_var": cvar_components["var"].mean(),
            "oracle_objective": regret_components["oracle_objective"].mean(),
        }
        return total_loss, mu_hat, weights, state, target_mu, details

    def _validation_loss(self, loader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                _, _, _, _, _, details = self._forward_batch(batch)
                losses.append(float(details["total_loss"].item()))
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
            train_losses = []
            prediction_losses = []
            utility_losses = []
            cvar_losses = []
            regret_losses = []
            for batch in train_loader:
                model_optimizer.zero_grad(set_to_none=True)
                loss, _, _, _, _, details = self._forward_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                model_optimizer.step()
                train_losses.append(float(loss.item()))
                prediction_losses.append(float(details["prediction_loss"].item()))
                utility_losses.append(float(details["utility_loss"].item()))
                cvar_losses.append(float(details["cvar_loss"].item()))
                regret_losses.append(float(details["regret_loss"].item()))

            val_loss = self._validation_loss(val_loader)
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
    def _decode_dates(value) -> List[str]:
        """Decode default-collated date strings for test batch size one."""

        if isinstance(value, str):
            return [value]
        if isinstance(value, (tuple, list)):
            if len(value) == 0:
                return []
            if all(isinstance(item, str) for item in value):
                return list(value)
            # DataLoader collates a list of H strings into H one-item tuples.
            if all(isinstance(item, (tuple, list)) for item in value):
                return [item[0] for item in value]
        return [str(value)]

    @staticmethod
    def _portfolio_metrics(daily_returns: pd.Series) -> Dict[str, float]:
        daily_returns = daily_returns.dropna().sort_index()
        if daily_returns.empty:
            raise RuntimeError("no daily portfolio returns were generated")
        equity = (1.0 + daily_returns).cumprod()
        annual_factor = 252.0
        volatility = float(daily_returns.std(ddof=1) * math.sqrt(annual_factor))
        sharpe = float(
            daily_returns.mean() / (daily_returns.std(ddof=1) + 1e-12)
            * math.sqrt(annual_factor)
        )
        years = len(daily_returns) / annual_factor
        terminal = float(equity.iloc[-1])
        annual_return = float(terminal ** (1.0 / years) - 1.0) if terminal > 0 else math.nan
        drawdown = equity / equity.cummax() - 1.0
        return {
            "TotalReturn": float(terminal - 1.0),
            "AnnualReturn": annual_return,
            "AnnualVolatility": volatility,
            "Sharpe": sharpe,
            "MaxDrawdown": float(drawdown.min()),
            "NumDailyObservations": float(len(daily_returns)),
        }

    def eval(self, setting, load=True):
        if load:
            checkpoint = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            if checkpoint.exists():
                self.model.load_state_dict(
                    torch.load(checkpoint, map_location=self.device, weights_only=True)
                )
        self.model.eval()
        _, test_loader = self._get_data("test")

        position_rows = []
        daily_return_rows: List[Tuple[str, float]] = []
        prediction_losses = []
        turnover_values = []
        budget_errors = []
        lower_violations = []
        upper_violations = []
        total_losses = []
        utility_losses = []
        cvar_losses = []
        regret_losses = []
        asset_names = [f"asset_{i}" for i in range(self.args.data_pool)]

        with torch.no_grad():
            for batch in test_loader:
                _, _, weights, state, _, details = self._forward_batch(batch)
                total_losses.append(float(details["total_loss"].item()))
                prediction_losses.append(float(details["prediction_loss"].item()))
                utility_losses.append(float(details["utility_loss"].item()))
                cvar_losses.append(float(details["cvar_loss"].item()))
                regret_losses.append(float(details["regret_loss"].item()))
                weights_np = weights.detach().cpu().numpy()
                future_np = batch["future_returns"].numpy()
                previous_np = batch["w_prev"].numpy()
                decision_dates = self._decode_dates(batch["decision_date"])
                future_dates = batch["future_dates"]

                budget_errors.extend(
                    state["budget_residual"].detach().cpu().numpy().reshape(-1).tolist()
                )
                lower_violations.append(
                    float(state["lower_violation"].detach().cpu().numpy().max())
                )
                upper_violations.append(
                    float(state["upper_violation"].detach().cpu().numpy().max())
                )

                for row_index, weight in enumerate(weights_np):
                    decision_date = decision_dates[row_index]
                    row = {"Date": decision_date}
                    row.update({name: float(value) for name, value in zip(asset_names, weight)})
                    position_rows.append(row)
                    turnover_values.append(
                        float(np.abs(weight - previous_np[row_index]).sum())
                    )

                    if isinstance(future_dates, list) and future_dates and isinstance(
                        future_dates[0], (tuple, list)
                    ):
                        dates_for_sample = [item[row_index] for item in future_dates]
                    else:
                        dates_for_sample = self._decode_dates(future_dates)
                    realized = future_np[row_index].dot(weight)
                    for date, value in zip(dates_for_sample, realized):
                        daily_return_rows.append((str(date), float(value)))

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

        metrics = self._portfolio_metrics(daily_series)
        metrics.update(
            {
                "LossMode": self.loss_mode,
                "TestLoss": float(np.mean(total_losses)),
                "PredictionLoss": float(np.mean(prediction_losses)),
                "UtilityLoss": float(np.mean(utility_losses)),
                "CVaRLoss": float(np.mean(cvar_losses)),
                "DecisionRegret": float(np.mean(regret_losses)),
                "MeanTurnover": float(np.mean(turnover_values)),
                "MaxAbsBudgetResidual": float(np.max(np.abs(budget_errors))),
                "MaxLowerViolation": float(np.max(lower_violations)),
                "MaxUpperViolation": float(np.max(upper_violations)),
            }
        )
        pd.DataFrame([metrics]).to_csv(result_dir / "test_metrics.csv", index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(daily["Date"], daily["Equity"], label="KKTFormer-v0")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / "test_equity_curve.png", dpi=150)
        plt.close()
        print(f"[Test] results saved to {result_dir}")
        print(metrics)
        return metrics
