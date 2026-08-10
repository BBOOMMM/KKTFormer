"""Stage 1: definition of the minimal KKTFormer portfolio problem.

This module deliberately does not solve the optimization problem.  It defines
the shapes, conventions, constraints, and objective that the differentiable
optimizer in a later stage must implement.

The first KKTFormer version makes one portfolio decision per sample.  The
model therefore produces ``mu_hat`` with shape ``(..., N)``.  A future return
path has shape ``(..., H, N)`` and is reduced to a per-asset target using the
configured aggregation rule.  ``H`` is the evaluation horizon, not a second
portfolio-decision axis.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Sequence, Tuple, Union

import torch


UpperBound = Union[Real, Sequence[Real]]


@dataclass(frozen=True)
class MinimalPortfolioProblem:
    """Specification for the first long-only KKTFormer decision problem.

    The mathematical problem is

    .. math::

        \min_w\; 0.5 w^T(\Sigma_t + \eta I)w - \hat\mu_t^T w

    subject to ``sum(w) == 1`` and ``0 <= w_i <= upper_bound_i``.

    ``return_aggregation='mean'`` is the v0 convention: the target for an
    asset is its arithmetic mean simple return over the next ``horizon``
    observations.  The optimizer itself is intentionally deferred to a later
    implementation stage.
    """

    num_assets: int
    lookback_window: int = 60
    horizon: int = 20
    rebalance_frequency: int = 20
    eta: float = 1e-3
    upper_bound: UpperBound = 1.0
    lower_bound: UpperBound = 0.0
    budget_target: float = 1.0
    return_aggregation: str = "mean"

    def __post_init__(self) -> None:
        if not isinstance(self.num_assets, int) or self.num_assets <= 0:
            raise ValueError("num_assets must be a positive integer")
        if not isinstance(self.lookback_window, int) or self.lookback_window <= 0:
            raise ValueError("lookback_window must be a positive integer")
        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if (
            not isinstance(self.rebalance_frequency, int)
            or self.rebalance_frequency <= 0
        ):
            raise ValueError("rebalance_frequency must be a positive integer")
        if not isinstance(self.eta, Real) or self.eta <= 0:
            raise ValueError("eta must be positive for a strongly convex problem")
        if not isinstance(self.budget_target, Real) or not torch.isfinite(
            torch.tensor(float(self.budget_target))
        ):
            raise ValueError("budget_target must be finite")
        if self.return_aggregation not in {"mean", "sum", "compound"}:
            raise ValueError(
                "return_aggregation must be one of: mean, sum, compound"
            )

        bounds = self._normalise_upper_bound(self.upper_bound)
        lower_bounds = self._normalise_upper_bound(self.lower_bound)
        if any(not torch.isfinite(torch.tensor(bound)) for bound in bounds):
            raise ValueError("all upper bounds must be finite")
        if any(not torch.isfinite(torch.tensor(bound)) for bound in lower_bounds):
            raise ValueError("all lower bounds must be finite")
        if any(lower >= upper for lower, upper in zip(lower_bounds, bounds)):
            raise ValueError("each lower bound must be smaller than its upper bound")
        target = float(self.budget_target)
        if sum(lower_bounds) > target or sum(bounds) < target:
            raise ValueError(
                "infeasible weight limits for the configured budget_target"
            )
        object.__setattr__(self, "upper_bound", bounds)
        object.__setattr__(self, "lower_bound", lower_bounds)

    def _normalise_upper_bound(self, upper_bound: UpperBound) -> Tuple[float, ...]:
        if isinstance(upper_bound, Real):
            return (float(upper_bound),) * self.num_assets

        try:
            bounds = tuple(float(value) for value in upper_bound)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "upper_bound must be a positive scalar or a sequence of length "
                "num_assets"
            ) from exc
        if len(bounds) != self.num_assets:
            raise ValueError(
                "a per-asset upper_bound sequence must have length num_assets"
            )
        return bounds

    @property
    def upper_bounds(self) -> Tuple[float, ...]:
        """Return the validated per-asset upper bounds."""

        # ``__post_init__`` converts the public field to a tuple.
        return self.upper_bound  # type: ignore[return-value]

    @property
    def lower_bounds(self) -> Tuple[float, ...]:
        """Return the validated per-asset lower bounds."""

        return self.lower_bound  # type: ignore[return-value]

    @property
    def decision_output_shape(self) -> Tuple[int]:
        """Shape of one model decision, excluding any batch dimensions."""

        return (self.num_assets,)

    @property
    def future_path_shape(self) -> Tuple[int, int]:
        """Shape of one future evaluation path, excluding batch dimensions."""

        return (self.horizon, self.num_assets)

    def aggregate_future_returns(self, future_returns: torch.Tensor) -> torch.Tensor:
        """Reduce a future return path to one target per asset.

        Args:
            future_returns: Tensor with shape ``(..., H, N)``.

        Returns:
            Tensor with shape ``(..., N)``.
        """

        if not isinstance(future_returns, torch.Tensor):
            raise TypeError("future_returns must be a torch.Tensor")
        if future_returns.ndim < 2:
            raise ValueError("future_returns must have shape (..., horizon, num_assets)")
        if future_returns.shape[-2:] != (self.horizon, self.num_assets):
            raise ValueError(
                "future_returns must have trailing shape "
                f"({self.horizon}, {self.num_assets}), got "
                f"{tuple(future_returns.shape[-2:])}"
            )

        if self.return_aggregation == "mean":
            return future_returns.mean(dim=-2)
        if self.return_aggregation == "sum":
            return future_returns.sum(dim=-2)
        return torch.prod(1.0 + future_returns, dim=-2) - 1.0

    def validate_mu(self, mu_hat: torch.Tensor) -> None:
        """Validate a model signal with shape ``(..., num_assets)``."""

        if not isinstance(mu_hat, torch.Tensor):
            raise TypeError("mu_hat must be a torch.Tensor")
        if mu_hat.ndim < 1 or mu_hat.shape[-1] != self.num_assets:
            raise ValueError(
                f"mu_hat must have trailing shape ({self.num_assets},), got "
                f"{tuple(mu_hat.shape)}"
            )

    def validate_sigma(self, sigma: torch.Tensor, atol: float = 1e-5) -> None:
        """Validate a covariance tensor with shape ``(N, N)`` or ``(..., N, N)``."""

        if not isinstance(sigma, torch.Tensor):
            raise TypeError("sigma must be a torch.Tensor")
        if sigma.ndim < 2 or sigma.shape[-2:] != (
            self.num_assets,
            self.num_assets,
        ):
            raise ValueError(
                "sigma must have trailing shape "
                f"({self.num_assets}, {self.num_assets}), got "
                f"{tuple(sigma.shape)}"
            )
        if not torch.allclose(sigma, sigma.transpose(-1, -2), atol=atol, rtol=0):
            raise ValueError("sigma must be symmetric")

    def constraint_residuals(self, weights: torch.Tensor) -> dict:
        """Return residuals for the v0 budget and box constraints.

        The returned values are useful for diagnostics and tests.  They do not
        project or repair the weights.
        """

        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")
        if weights.ndim < 1 or weights.shape[-1] != self.num_assets:
            raise ValueError(
                f"weights must have trailing shape ({self.num_assets},), got "
                f"{tuple(weights.shape)}"
            )
        lower = torch.as_tensor(
            self.lower_bounds, dtype=weights.dtype, device=weights.device
        )
        upper = torch.as_tensor(
            self.upper_bounds, dtype=weights.dtype, device=weights.device
        )
        return {
            "budget": weights.sum(dim=-1) - self.budget_target,
            "lower_violation": torch.relu(lower - weights),
            "upper_violation": torch.relu(weights - upper),
        }

    def is_feasible(self, weights: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
        """Return a boolean feasibility mask for one or more portfolios."""

        residuals = self.constraint_residuals(weights)
        budget_ok = residuals["budget"].abs() <= atol
        lower_ok = residuals["lower_violation"].amax(dim=-1) <= atol
        upper_ok = residuals["upper_violation"].amax(dim=-1) <= atol
        return budget_ok & lower_ok & upper_ok

    def objective(
        self,
        weights: torch.Tensor,
        mu_hat: torch.Tensor,
        sigma: torch.Tensor,
        w_prev: torch.Tensor = None,
        turnover_penalty: float = 0.0,
        transaction_cost_rate: torch.Tensor = None,
        transaction_cost_smoothing: float = 1e-6,
    ) -> torch.Tensor:
        """Evaluate the v0 quadratic objective without solving it.

        Broadcasting is supported for a shared ``sigma`` with shape ``(N, N)``
        or per-sample covariance with shape ``(..., N, N)``.
        """

        self.validate_mu(mu_hat)
        self.validate_sigma(sigma)
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")
        if weights.shape != mu_hat.shape:
            raise ValueError(
                f"weights and mu_hat must have identical shape, got "
                f"{tuple(weights.shape)} and {tuple(mu_hat.shape)}"
            )
        if weights.shape[-1] != self.num_assets:
            raise ValueError(
                f"weights must have trailing shape ({self.num_assets},), got "
                f"{tuple(weights.shape)}"
            )

        eye = torch.eye(
            self.num_assets, dtype=sigma.dtype, device=sigma.device
        )
        q = sigma + self.eta * eye
        if q.ndim == 2 and weights.ndim > 1:
            q = q.expand(weights.shape[:-1] + q.shape)
        risk = 0.5 * torch.einsum("...i,...ij,...j->...", weights, q, weights)
        linear = -(weights * mu_hat).sum(dim=-1)
        objective = risk + linear
        if turnover_penalty < 0:
            raise ValueError("turnover_penalty cannot be negative")
        if transaction_cost_smoothing <= 0:
            raise ValueError("transaction_cost_smoothing must be positive")
        if w_prev is not None:
            if w_prev.shape != weights.shape:
                raise ValueError("w_prev must have the same shape as weights")
            delta = weights - w_prev
            objective = objective + 0.5 * float(turnover_penalty) * delta.square().sum(dim=-1)
            if transaction_cost_rate is not None:
                rate = torch.as_tensor(
                    transaction_cost_rate, dtype=weights.dtype, device=weights.device
                )
                while rate.ndim < weights.ndim - 1:
                    rate = rate.unsqueeze(-1)
                smooth_l1 = (
                    torch.sqrt(delta.square() + transaction_cost_smoothing**2)
                    - transaction_cost_smoothing
                ).sum(dim=-1)
                objective = objective + rate * smooth_l1
        elif float(turnover_penalty) != 0.0 or transaction_cost_rate is not None:
            raise ValueError("w_prev is required for turnover and transaction cost")
        return objective
