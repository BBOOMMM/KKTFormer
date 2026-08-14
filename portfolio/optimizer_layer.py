"""Differentiable portfolio optimization with trading and risk constraints.

The layer solves the strongly-convex quadratic portfolio problem with a fixed
number of projected-gradient iterations.  In addition to the original
budget/box constraints, stage 7 adds:

* a quadratic turnover penalty and a smooth transaction-cost term in the
  optimized objective;
* linear factor and industry exposure bounds;
* an L1 turnover cap and an optional gross-exposure cap.

The feasible-set projection uses Dykstra iterations over bounded-simplex,
half-space and L1-ball projections.  All iterations are fixed and implemented
with PyTorch operations, so the layer remains usable inside training.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from portfolio.problem import MinimalPortfolioProblem


class RiskBudgetedAllocator(nn.Module):
    """Differentiable softmax allocator with a turnover-aware proximal step.

    The allocation logits are allowed to contain a risk-budget prior (for
    example a multi-scale trend/volatility score) and a learned residual.  A
    plain softmax treats every logit as a pure alpha score; this layer adds the
    missing portfolio decision geometry by shrinking a proposed allocation
    towards the current holdings when trading is expensive or a turnover cap
    is active.  The shrinkage is smooth and keeps the unit-simplex invariant.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        turnover_aversion: float = 0.0,
        entropy_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if temperature <= 0 or not math.isfinite(float(temperature)):
            raise ValueError("temperature must be finite and positive")
        if turnover_aversion < 0 or not math.isfinite(float(turnover_aversion)):
            raise ValueError("turnover_aversion must be finite and non-negative")
        if entropy_epsilon <= 0 or not math.isfinite(float(entropy_epsilon)):
            raise ValueError("entropy_epsilon must be finite and positive")
        self.temperature = float(temperature)
        self.turnover_aversion = float(turnover_aversion)
        self.entropy_epsilon = float(entropy_epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        w_prev: Optional[torch.Tensor] = None,
        max_turnover=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.ndim < 1:
            raise ValueError("logits must have at least one dimension")
        proposal = torch.softmax(logits / self.temperature, dim=-1)
        if w_prev is None:
            weights = proposal
            turnover = torch.zeros_like(proposal[..., 0])
        else:
            if w_prev.shape != proposal.shape:
                raise ValueError("w_prev must have the same shape as logits")
            turnover = (proposal - w_prev).abs().sum(dim=-1)
            # This is the closed-form proximal blend for a quadratic turnover
            # aversion.  It is intentionally bounded in [0, 1] and therefore
            # cannot destabilize the simplex policy.
            blend = 1.0 / (1.0 + self.turnover_aversion)
            if max_turnover is not None:
                cap = torch.as_tensor(
                    max_turnover, dtype=proposal.dtype, device=proposal.device
                )
                while cap.ndim < proposal.ndim - 1:
                    cap = cap.unsqueeze(-1)
                cap = cap.clamp_min(1e-6)
                cap_blend = torch.minimum(
                    torch.ones_like(turnover), cap.squeeze(-1) / turnover.clamp_min(1e-6)
                )
                blend = blend * cap_blend
            weights = w_prev + blend * (proposal - w_prev)
            turnover = (weights - w_prev).abs().sum(dim=-1)
        state = {
            "proposal_turnover": turnover,
            "proposal_entropy": (
                -proposal * torch.log(proposal.clamp_min(self.entropy_epsilon))
            ).sum(dim=-1),
            "turnover_shrinkage": (
                (weights - proposal).abs().sum(dim=-1)
                if w_prev is not None
                else torch.zeros_like(turnover)
            ),
        }
        return weights, state


def _as_batch_scalar(
    value,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
    default: float = 0.0,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), default, dtype=dtype, device=device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    elif tensor.ndim == 1 and tensor.shape[0] == batch_size:
        pass
    else:
        raise ValueError(f"{name} must be scalar or have shape (batch_size,)")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return tensor


class _CappedSimplexProjection(torch.autograd.Function):
    """Projection onto a simplex with per-coordinate upper bounds.

    The backward is the exact local active-set Jacobian away from a bound
    transition.  ``target`` is a tensor so lower-bound shifts can have a
    different residual target for every sample.
    """

    @staticmethod
    def forward(ctx, values, upper, target, bisection_steps):
        with torch.no_grad():
            target = target.to(dtype=values.dtype, device=values.device)
            while target.ndim < values.ndim - 1:
                target = target.unsqueeze(-1)
            tau_low = (values - upper).amin(dim=-1)
            tau_high = values.amax(dim=-1)
            for _ in range(int(bisection_steps)):
                tau_mid = 0.5 * (tau_low + tau_high)
                projected = torch.clamp(values - tau_mid.unsqueeze(-1), min=0.0)
                projected = torch.minimum(projected, upper)
                mass_too_high = projected.sum(dim=-1) > target.squeeze(-1)
                tau_low = torch.where(mass_too_high, tau_mid, tau_low)
                tau_high = torch.where(mass_too_high, tau_high, tau_mid)

            tau = 0.5 * (tau_low + tau_high)
            projected = torch.clamp(values - tau.unsqueeze(-1), min=0.0)
            projected = torch.minimum(projected, upper)
            tolerance = 1e-7 if values.dtype == torch.float64 else 1e-5
            free = (projected > tolerance) & (projected < upper - tolerance)

            residual = target - projected.sum(dim=-1, keepdim=True)
            free_count = free.sum(dim=-1, keepdim=True).to(values.dtype)
            projected = projected + free.to(values.dtype) * (
                residual / free_count.clamp_min(1.0)
            )
            projected = torch.clamp(projected, min=0.0)
            projected = torch.minimum(projected, upper)
            free = (projected > tolerance) & (projected < upper - tolerance)

        ctx.save_for_backward(free)
        return projected

    @staticmethod
    def backward(ctx, grad_output):
        (free,) = ctx.saved_tensors
        if grad_output is None:
            return None, None, None, None
        free_count = free.sum(dim=-1, keepdim=True).to(grad_output.dtype)
        free_grad = torch.where(free, grad_output, torch.zeros_like(grad_output))
        free_mean = free_grad.sum(dim=-1, keepdim=True) / free_count.clamp_min(1.0)
        grad_values = torch.where(
            free,
            grad_output - free_mean,
            torch.zeros_like(grad_output),
        )
        return grad_values, None, None, None


def project_capped_simplex(
    values: torch.Tensor,
    upper_bounds: torch.Tensor,
    target=1.0,
    bisection_steps: int = 64,
) -> torch.Tensor:
    """Project vectors onto ``0 <= w <= upper`` and ``sum(w)=target``."""

    if not isinstance(values, torch.Tensor) or not isinstance(upper_bounds, torch.Tensor):
        raise TypeError("values and upper_bounds must be torch.Tensor objects")
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if upper_bounds.ndim not in (1, values.ndim):
        raise ValueError("upper_bounds must have shape (N,) or match values rank")
    if upper_bounds.shape[-1] != values.shape[-1]:
        raise ValueError("upper_bounds and values must have the same asset dimension")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")
    if not torch.isfinite(values).all() or not torch.isfinite(upper_bounds).all():
        raise ValueError("values and upper_bounds must be finite")
    if (upper_bounds <= 0).any():
        raise ValueError("upper_bounds must be positive")

    upper = upper_bounds.expand_as(values) if upper_bounds.ndim == 1 else upper_bounds
    if upper.shape != values.shape:
        raise ValueError("batched upper_bounds must match values shape")
    target_tensor = torch.as_tensor(target, dtype=values.dtype, device=values.device)
    if target_tensor.ndim == values.ndim - 1:
        target_tensor = target_tensor.unsqueeze(-1)
    while target_tensor.ndim < values.ndim - 1:
        target_tensor = target_tensor.unsqueeze(-1)
    if (upper.sum(dim=-1, keepdim=True) < target_tensor).any():
        raise ValueError("upper_bounds do not permit the requested target sum")
    if (target_tensor < 0).any():
        raise ValueError("target must be non-negative for a capped simplex")
    return _CappedSimplexProjection.apply(
        values, upper, target_tensor, int(bisection_steps)
    )


def project_bounded_simplex(
    values: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    target=1.0,
    bisection_steps: int = 64,
) -> torch.Tensor:
    """Project onto ``lower <= w <= upper`` and ``sum(w)=target``."""

    if lower_bounds.shape != upper_bounds.shape and lower_bounds.ndim != 1:
        raise ValueError("lower_bounds and upper_bounds have incompatible shapes")
    lower = lower_bounds.expand_as(values) if lower_bounds.ndim == 1 else lower_bounds
    upper = upper_bounds.expand_as(values) if upper_bounds.ndim == 1 else upper_bounds
    if lower.shape != values.shape or upper.shape != values.shape:
        raise ValueError("bounds must have shape (N,) or match values")
    if (lower >= upper).any():
        raise ValueError("lower_bounds must be smaller than upper_bounds")
    target_tensor = torch.as_tensor(target, dtype=values.dtype, device=values.device)
    if target_tensor.ndim == values.ndim - 1:
        target_tensor = target_tensor.unsqueeze(-1)
    while target_tensor.ndim < values.ndim - 1:
        target_tensor = target_tensor.unsqueeze(-1)
    shifted_target = target_tensor - lower.sum(dim=-1, keepdim=True)
    return lower + project_capped_simplex(
        values - lower,
        upper - lower,
        target=shifted_target,
        bisection_steps=bisection_steps,
    )


def project_l1_ball(values: torch.Tensor, radius) -> torch.Tensor:
    """Project each row onto an L1 ball with a possibly batched radius."""

    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    radius_tensor = torch.as_tensor(radius, dtype=values.dtype, device=values.device)
    while radius_tensor.ndim < values.ndim:
        radius_tensor = radius_tensor.unsqueeze(-1)
    if (radius_tensor < 0).any() or not torch.isfinite(radius_tensor).all():
        raise ValueError("L1 radius must be finite and non-negative")

    absolute = values.abs()
    inside = absolute.sum(dim=-1, keepdim=True) <= radius_tensor
    sorted_abs, _ = torch.sort(absolute, dim=-1, descending=True)
    cumulative = sorted_abs.cumsum(dim=-1)
    ranks = torch.arange(
        1,
        values.shape[-1] + 1,
        dtype=values.dtype,
        device=values.device,
    )
    threshold_candidates = (cumulative - radius_tensor) / ranks
    active = sorted_abs > threshold_candidates
    rho = active.sum(dim=-1).clamp_min(1).long() - 1
    threshold = threshold_candidates.gather(-1, rho.unsqueeze(-1))
    projected = values.sign() * torch.relu(absolute - threshold)
    return torch.where(inside, values, projected)


def _project_upper_halfspace(values, coefficients, upper):
    exposure = (values * coefficients).sum(dim=-1)
    violation = torch.relu(exposure - upper)
    norm_sq = coefficients.square().sum(dim=-1).clamp_min(1e-12)
    return values - (violation / norm_sq).unsqueeze(-1) * coefficients


def _project_lower_halfspace(values, coefficients, lower):
    exposure = (values * coefficients).sum(dim=-1)
    violation = torch.relu(lower - exposure)
    norm_sq = coefficients.square().sum(dim=-1).clamp_min(1e-12)
    return values + (violation / norm_sq).unsqueeze(-1) * coefficients


def _has_bound(bound) -> bool:
    return bound is not None


class DifferentiablePortfolioOptimizer(nn.Module):
    """Projected-gradient solver for the stage-7 constrained QP."""

    def __init__(
        self,
        problem: MinimalPortfolioProblem,
        num_iterations: int = 200,
        bisection_steps: int = 64,
        step_size: Optional[float] = None,
        constraint_projection_iterations: int = 20,
        entropy_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if bisection_steps <= 0 or constraint_projection_iterations <= 0:
            raise ValueError("projection iteration counts must be positive")
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive")
        if not math.isfinite(float(entropy_epsilon)) or entropy_epsilon <= 0:
            raise ValueError("entropy_epsilon must be finite and positive")
        self.problem = problem
        self.num_iterations = int(num_iterations)
        self.bisection_steps = int(bisection_steps)
        self.step_size = None if step_size is None else float(step_size)
        self.constraint_projection_iterations = int(constraint_projection_iterations)
        self.entropy_epsilon = float(entropy_epsilon)

    def _flatten_asset_tensor(
        self,
        tensor: Optional[torch.Tensor],
        name: str,
        batch_size: int,
        num_assets: int,
        original_batch_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        tensor = torch.as_tensor(tensor, dtype=dtype, device=device)
        if tensor.ndim == 1 and tensor.shape[0] == num_assets:
            result = tensor.unsqueeze(0).expand(batch_size, -1)
        elif tuple(tensor.shape[:-1]) == original_batch_shape and tensor.shape[-1] == num_assets:
            result = tensor.reshape(batch_size, num_assets)
        elif tensor.ndim == 2 and tensor.shape == (batch_size, num_assets):
            result = tensor
        else:
            raise ValueError(
                f"{name} must have shape (N,), (B, N), or the same leading batch shape as mu_hat"
            )
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return result

    def _flatten_exposure(
        self,
        exposure: Optional[torch.Tensor],
        name: str,
        batch_size: int,
        num_assets: int,
        original_batch_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if exposure is None:
            return None
        exposure = torch.as_tensor(exposure, dtype=dtype, device=device)
        if exposure.ndim == 2 and exposure.shape[0] == num_assets:
            result = exposure.unsqueeze(0).expand(batch_size, -1, -1)
        elif exposure.ndim == 3 and tuple(exposure.shape[:-2]) == original_batch_shape:
            result = exposure.reshape(batch_size, num_assets, exposure.shape[-1])
        elif exposure.ndim == 3 and exposure.shape[0] == batch_size and exposure.shape[1] == num_assets:
            result = exposure
        else:
            raise ValueError(f"{name} must have shape (N, K) or (B, N, K)")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return result

    def _flatten_exposure_bound(
        self,
        bound,
        name: str,
        batch_size: int,
        num_factors: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if bound is None:
            return None
        bound = torch.as_tensor(bound, dtype=dtype, device=device)
        if bound.ndim == 0:
            result = bound.expand(batch_size, num_factors)
        elif bound.ndim == 1 and bound.shape[0] == num_factors:
            result = bound.unsqueeze(0).expand(batch_size, -1)
        elif bound.ndim == 2 and bound.shape == (batch_size, num_factors):
            result = bound
        else:
            raise ValueError(f"{name} must be scalar, (K,), or (B, K)")
        if torch.isnan(result).any():
            raise ValueError(f"{name} contains NaN")
        return result

    def _prepare_inputs(
        self,
        mu_hat,
        sigma,
        factor_exposure,
        industry_exposure,
        w_prev,
        upper_bounds,
        lower_bounds,
        initial_weights,
        factor_lower,
        factor_upper,
        industry_lower,
        industry_upper,
        max_turnover,
        gross_exposure_limit,
    ):
        self.problem.validate_mu(mu_hat)
        self.problem.validate_sigma(sigma)
        original_batch_shape = tuple(mu_hat.shape[:-1])
        batch_size = int(mu_hat.numel() // self.problem.num_assets)
        num_assets = self.problem.num_assets
        mu = mu_hat.reshape(batch_size, num_assets)
        if sigma.ndim == 2:
            sigma_batch = sigma.unsqueeze(0).expand(batch_size, num_assets, num_assets)
        elif tuple(sigma.shape[:-2]) == original_batch_shape:
            sigma_batch = sigma.reshape(batch_size, num_assets, num_assets)
        elif sigma.ndim == 3 and sigma.shape[0] == batch_size:
            sigma_batch = sigma
        else:
            raise ValueError("sigma must have shape (N,N), (B,N,N), or match mu_hat batch shape")

        lower = self._flatten_asset_tensor(
            lower_bounds, "lower_bounds", batch_size, num_assets,
            original_batch_shape, mu.dtype, mu.device
        )
        upper = self._flatten_asset_tensor(
            upper_bounds, "upper_bounds", batch_size, num_assets,
            original_batch_shape, mu.dtype, mu.device
        )
        if lower is None:
            lower = torch.as_tensor(self.problem.lower_bounds, dtype=mu.dtype, device=mu.device)
            lower = lower.unsqueeze(0).expand(batch_size, -1)
        if upper is None:
            upper = torch.as_tensor(self.problem.upper_bounds, dtype=mu.dtype, device=mu.device)
            upper = upper.unsqueeze(0).expand(batch_size, -1)
        if (lower >= upper).any():
            raise ValueError("lower_bounds must be smaller than upper_bounds")

        previous = self._flatten_asset_tensor(
            w_prev, "w_prev", batch_size, num_assets,
            original_batch_shape, mu.dtype, mu.device
        )
        initial = self._flatten_asset_tensor(
            initial_weights, "initial_weights", batch_size, num_assets,
            original_batch_shape, mu.dtype, mu.device
        )
        factor = self._flatten_exposure(
            factor_exposure, "factor_exposure", batch_size, num_assets,
            original_batch_shape, mu.dtype, mu.device
        )
        industry = self._flatten_exposure(
            industry_exposure, "industry_exposure", batch_size, num_assets,
            original_batch_shape, mu.dtype, mu.device
        )
        factor_l = self._flatten_exposure_bound(
            factor_lower, "factor_lower", batch_size,
            factor.shape[-1] if factor is not None else 0, mu.dtype, mu.device
        ) if factor is not None else None
        factor_u = self._flatten_exposure_bound(
            factor_upper, "factor_upper", batch_size,
            factor.shape[-1] if factor is not None else 0, mu.dtype, mu.device
        ) if factor is not None else None
        industry_l = self._flatten_exposure_bound(
            industry_lower, "industry_lower", batch_size,
            industry.shape[-1] if industry is not None else 0, mu.dtype, mu.device
        ) if industry is not None else None
        industry_u = self._flatten_exposure_bound(
            industry_upper, "industry_upper", batch_size,
            industry.shape[-1] if industry is not None else 0, mu.dtype, mu.device
        ) if industry is not None else None
        turnover_limit = _as_batch_scalar(
            max_turnover, batch_size, mu.dtype, mu.device, "max_turnover", default=float("inf")
        ) if max_turnover is not None else None
        gross_limit = _as_batch_scalar(
            gross_exposure_limit, batch_size, mu.dtype, mu.device,
            "gross_exposure_limit", default=float("inf")
        ) if gross_exposure_limit is not None else None
        return (
            mu, sigma_batch, lower, upper, previous, initial, factor, industry,
            factor_l, factor_u, industry_l, industry_u, turnover_limit, gross_limit,
            original_batch_shape,
        )

    def _project_feasible(
        self,
        values,
        lower,
        upper,
        factor,
        factor_lower,
        factor_upper,
        industry,
        industry_lower,
        industry_upper,
        previous,
        max_turnover,
        gross_limit,
    ):
        target = self.problem.budget_target
        has_linear = factor is not None and (factor_lower is not None or factor_upper is not None)
        has_industry = industry is not None and (industry_lower is not None or industry_upper is not None)
        has_turnover_cap = previous is not None and max_turnover is not None
        has_gross_cap = gross_limit is not None
        if not (has_linear or has_industry or has_turnover_cap or has_gross_cap):
            return project_bounded_simplex(
                values, lower, upper, target=target, bisection_steps=self.bisection_steps
            )

        z = project_bounded_simplex(
            values, lower, upper, target=target, bisection_steps=self.bisection_steps
        )
        num_sets = 1
        if has_linear:
            num_sets += int(factor.shape[-1]) * int(factor_lower is not None) + int(factor.shape[-1]) * int(factor_upper is not None)
        if has_industry:
            num_sets += int(industry.shape[-1]) * int(industry_lower is not None) + int(industry.shape[-1]) * int(industry_upper is not None)
        if has_turnover_cap:
            num_sets += 1
        if has_gross_cap:
            num_sets += 1
        corrections = torch.zeros(
            z.shape[0], num_sets, z.shape[-1], dtype=z.dtype, device=z.device
        )
        for _ in range(self.constraint_projection_iterations):
            set_index = 0
            shifted = z + corrections[:, set_index]
            projected = project_bounded_simplex(
                shifted, lower, upper, target=target, bisection_steps=self.bisection_steps
            )
            corrections[:, set_index] = shifted - projected
            z = projected
            set_index += 1

            def apply_halfspaces(exposure, lower_bound, upper_bound):
                nonlocal z, set_index
                if exposure is None:
                    return
                for factor_index in range(exposure.shape[-1]):
                    coefficients = exposure[:, :, factor_index]
                    if lower_bound is not None:
                        shifted_local = z + corrections[:, set_index]
                        z = _project_lower_halfspace(
                            shifted_local, coefficients, lower_bound[:, factor_index]
                        )
                        corrections[:, set_index] = shifted_local - z
                        set_index += 1
                    if upper_bound is not None:
                        shifted_local = z + corrections[:, set_index]
                        z = _project_upper_halfspace(
                            shifted_local, coefficients, upper_bound[:, factor_index]
                        )
                        corrections[:, set_index] = shifted_local - z
                        set_index += 1

            if has_linear:
                apply_halfspaces(factor, factor_lower, factor_upper)
            if has_industry:
                apply_halfspaces(industry, industry_lower, industry_upper)
            if has_turnover_cap:
                shifted = z + corrections[:, set_index]
                z = previous + project_l1_ball(shifted - previous, max_turnover)
                corrections[:, set_index] = shifted - z
                set_index += 1
            if has_gross_cap:
                shifted = z + corrections[:, set_index]
                z = project_l1_ball(shifted, gross_limit)
                corrections[:, set_index] = shifted - z
        return z

    def _step_size_for(
        self,
        q,
        transaction_cost_rate,
        transaction_cost_smoothing,
        entropy_regularization=0.0,
    ):
        if self.step_size is not None:
            eigenvalues = torch.linalg.eigvalsh(q)
            if (eigenvalues[..., 0] <= 0).any():
                raise ValueError("Sigma + eta I + turnover penalty must be positive definite")
            return torch.full(
                (q.shape[0],), self.step_size, dtype=q.dtype, device=q.device
            )
        eigenvalues = torch.linalg.eigvalsh(q)
        if (eigenvalues[..., 0] <= 0).any():
            raise ValueError("Sigma + eta I + turnover penalty must be positive definite")
        smooth_lipschitz = transaction_cost_rate / transaction_cost_smoothing
        # The smoothed entropy gradient has local Lipschitz constant
        # ``tau / entropy_epsilon``.  Include it in the step-size bound so a
        # non-zero entropy coefficient does not destabilise the projected
        # gradient iterations near the lower boundary.
        entropy_lipschitz = float(entropy_regularization) / self.entropy_epsilon
        return 0.95 / (
            eigenvalues[..., -1] + smooth_lipschitz + entropy_lipschitz
        ).clamp_min(1e-8)

    def forward(
        self,
        mu_hat: torch.Tensor,
        sigma: torch.Tensor,
        factor_exposure: Optional[torch.Tensor] = None,
        w_prev: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
        initial_weights: Optional[torch.Tensor] = None,
        lower_bounds: Optional[torch.Tensor] = None,
        factor_lower=None,
        factor_upper=None,
        industry_exposure: Optional[torch.Tensor] = None,
        industry_lower=None,
        industry_upper=None,
        budget_target: Optional[float] = None,
        turnover_penalty: float = 0.0,
        transaction_cost_rate=None,
        transaction_cost_smoothing: float = 1e-4,
        max_turnover=None,
        gross_exposure_limit=None,
        entropy_regularization: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Solve the constrained portfolio problem and return diagnostics."""

        if turnover_penalty < 0:
            raise ValueError("turnover_penalty cannot be negative")
        entropy_regularization = float(entropy_regularization)
        if not math.isfinite(entropy_regularization) or entropy_regularization < 0:
            raise ValueError("entropy_regularization must be finite and non-negative")
        if entropy_regularization != 0.0 and min(self.problem.lower_bounds) < 0:
            raise ValueError(
                "entropy_regularization requires non-negative portfolio weights"
            )
        if transaction_cost_smoothing <= 0:
            raise ValueError("transaction_cost_smoothing must be positive")
        if budget_target is not None and float(budget_target) != self.problem.budget_target:
            raise ValueError("budget_target must match the MinimalPortfolioProblem")
        (
            mu, sigma_batch, lower, upper, previous, initial, factor, industry,
            factor_l, factor_u, industry_l, industry_u, turnover_limit, gross_limit,
            original_batch_shape,
        ) = self._prepare_inputs(
            mu_hat, sigma, factor_exposure, industry_exposure, w_prev,
            upper_bounds, lower_bounds, initial_weights, factor_lower, factor_upper,
            industry_lower, industry_upper, max_turnover, gross_exposure_limit,
        )
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma_batch).all():
            raise ValueError("mu_hat and sigma must be finite")

        n_assets = self.problem.num_assets
        eye = torch.eye(n_assets, dtype=sigma_batch.dtype, device=sigma_batch.device)
        q = 0.5 * (sigma_batch + sigma_batch.transpose(-1, -2))
        q = q + self.problem.eta * eye.unsqueeze(0)
        if previous is not None and turnover_penalty != 0.0:
            q = q + float(turnover_penalty) * eye.unsqueeze(0)
            effective_mu = mu + float(turnover_penalty) * previous
        else:
            effective_mu = mu

        cost_rate = _as_batch_scalar(
            transaction_cost_rate, mu.shape[0], mu.dtype, mu.device,
            "transaction_cost_rate", default=0.0
        )
        if previous is None and (turnover_penalty != 0.0 or (cost_rate != 0).any()):
            raise ValueError("w_prev is required for turnover penalty and transaction cost")
        step = self._step_size_for(
            q,
            cost_rate,
            transaction_cost_smoothing,
            entropy_regularization=entropy_regularization,
        )

        if initial is None:
            initial = previous
        if initial is None:
            initial = torch.zeros_like(mu)
        weights = self._project_feasible(
            initial, lower, upper, factor, factor_l, factor_u, industry,
            industry_l, industry_u, previous, turnover_limit, gross_limit,
        )

        last_step_norm = torch.zeros(mu.shape[0], dtype=mu.dtype, device=mu.device)
        for _ in range(self.num_iterations):
            gradient = torch.bmm(q, weights.unsqueeze(-1)).squeeze(-1) - effective_mu
            if entropy_regularization != 0.0:
                # ``w log w`` has an unbounded derivative at zero.  The
                # epsilon-smoothed form is numerically stable and keeps the
                # optimizer differentiable at the lower bound.
                safe_weights = weights + self.entropy_epsilon
                gradient = gradient + entropy_regularization * (
                    torch.log(safe_weights) + 1.0
                )
            if previous is not None and (cost_rate != 0).any():
                delta = weights - previous
                gradient = gradient + cost_rate.unsqueeze(-1) * delta / torch.sqrt(
                    delta.square() + transaction_cost_smoothing**2
                )
            candidate = weights - step.unsqueeze(-1) * gradient
            next_weights = self._project_feasible(
                candidate, lower, upper, factor, factor_l, factor_u, industry,
                industry_l, industry_u, previous, turnover_limit, gross_limit,
            )
            last_step_norm = (next_weights - weights).abs().amax(dim=-1)
            weights = next_weights

        delta = weights - previous if previous is not None else torch.zeros_like(weights)
        turnover = delta.abs().sum(dim=-1)
        smooth_transaction_cost = cost_rate * (
            torch.sqrt(delta.square() + transaction_cost_smoothing**2)
            - transaction_cost_smoothing
        ).sum(dim=-1)
        actual_transaction_cost = cost_rate * turnover
        objective = self.problem.objective(
            weights,
            mu,
            sigma_batch,
            w_prev=previous,
            turnover_penalty=turnover_penalty,
            transaction_cost_rate=cost_rate if previous is not None else None,
            transaction_cost_smoothing=transaction_cost_smoothing,
            entropy_regularization=entropy_regularization,
            entropy_epsilon=self.entropy_epsilon,
        )
        residuals = self.problem.constraint_residuals(weights)
        factor_value = (
            torch.einsum("bi,bik->bk", weights, factor)
            if factor is not None else None
        )
        industry_value = (
            torch.einsum("bi,bik->bk", weights, industry)
            if industry is not None else None
        )
        factor_lower_violation = (
            torch.relu(factor_l - factor_value) if factor_value is not None and factor_l is not None
            else torch.zeros(weights.shape[0], 0, dtype=weights.dtype, device=weights.device)
        )
        factor_upper_violation = (
            torch.relu(factor_value - factor_u) if factor_value is not None and factor_u is not None
            else torch.zeros(weights.shape[0], 0, dtype=weights.dtype, device=weights.device)
        )
        industry_lower_violation = (
            torch.relu(industry_l - industry_value) if industry_value is not None and industry_l is not None
            else torch.zeros(weights.shape[0], 0, dtype=weights.dtype, device=weights.device)
        )
        industry_upper_violation = (
            torch.relu(industry_value - industry_u) if industry_value is not None and industry_u is not None
            else torch.zeros(weights.shape[0], 0, dtype=weights.dtype, device=weights.device)
        )
        active_parts = [
            (weights - lower <= 1e-5).to(weights.dtype),
            (upper - weights <= 1e-5).to(weights.dtype),
        ]
        if factor_value is not None:
            if factor_l is not None:
                active_parts.append((factor_value - factor_l <= 1e-5).to(weights.dtype))
            if factor_u is not None:
                active_parts.append((factor_u - factor_value <= 1e-5).to(weights.dtype))
        if industry_value is not None:
            if industry_l is not None:
                active_parts.append((industry_value - industry_l <= 1e-5).to(weights.dtype))
            if industry_u is not None:
                active_parts.append((industry_u - industry_value <= 1e-5).to(weights.dtype))
        active_ratio = torch.cat([part.reshape(weights.shape[0], -1) for part in active_parts], dim=-1).mean(dim=-1)
        if turnover_limit is not None:
            active_ratio = active_ratio + (turnover_limit - turnover <= 1e-5).to(weights.dtype)
            active_ratio = active_ratio / 2.0
        if gross_limit is not None:
            active_ratio = active_ratio + (gross_limit - weights.abs().sum(dim=-1) <= 1e-5).to(weights.dtype)
            active_ratio = active_ratio / 2.0

        output_shape = original_batch_shape + (n_assets,)
        state: Dict[str, torch.Tensor] = {
            "objective": objective.reshape(original_batch_shape),
            "iterations": torch.as_tensor(self.num_iterations, dtype=torch.int64, device=mu.device),
            "last_step_norm": last_step_norm.reshape(original_batch_shape),
            "budget_residual": residuals["budget"].reshape(original_batch_shape),
            "lower_violation": residuals["lower_violation"].reshape(output_shape),
            "upper_violation": residuals["upper_violation"].reshape(output_shape),
            "turnover": turnover.reshape(original_batch_shape),
            "turnover_penalty": (0.5 * float(turnover_penalty) * delta.square().sum(dim=-1)).reshape(original_batch_shape),
            "transaction_cost": actual_transaction_cost.reshape(original_batch_shape),
            "smooth_transaction_cost": smooth_transaction_cost.reshape(original_batch_shape),
            "entropy": (-weights * torch.log(weights.clamp_min(self.entropy_epsilon))).sum(dim=-1).reshape(original_batch_shape),
            "entropy_penalty": (
                entropy_regularization
                * (
                    (weights + self.entropy_epsilon)
                    * torch.log(weights + self.entropy_epsilon)
                    - self.entropy_epsilon * math.log(self.entropy_epsilon)
                ).sum(dim=-1)
            ).reshape(original_batch_shape),
            "active_constraint_ratio": active_ratio.reshape(original_batch_shape),
            "net_exposure": weights.sum(dim=-1).reshape(original_batch_shape),
            "gross_exposure": weights.abs().sum(dim=-1).reshape(original_batch_shape),
        }
        if factor_value is not None:
            state.update({
                "factor_exposure": factor_value.reshape(original_batch_shape + (factor_value.shape[-1],)),
                "factor_lower_violation": factor_lower_violation.reshape(original_batch_shape + (factor_lower_violation.shape[-1],)),
                "factor_upper_violation": factor_upper_violation.reshape(original_batch_shape + (factor_upper_violation.shape[-1],)),
            })
        if industry_value is not None:
            state.update({
                "industry_exposure": industry_value.reshape(original_batch_shape + (industry_value.shape[-1],)),
                "industry_lower_violation": industry_lower_violation.reshape(original_batch_shape + (industry_lower_violation.shape[-1],)),
                "industry_upper_violation": industry_upper_violation.reshape(original_batch_shape + (industry_upper_violation.shape[-1],)),
            })
        if turnover_limit is not None:
            state["turnover_violation"] = torch.relu(turnover - turnover_limit).reshape(original_batch_shape)
        if gross_limit is not None:
            state["gross_exposure_violation"] = torch.relu(
                weights.abs().sum(dim=-1) - gross_limit
            ).reshape(original_batch_shape)
        return weights.reshape(output_shape), state
