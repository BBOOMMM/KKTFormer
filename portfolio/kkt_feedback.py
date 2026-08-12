"""KKT state extraction for the minimal KKTFormer quadratic program."""

import math
from typing import Dict

import torch

from portfolio.problem import MinimalPortfolioProblem


def _batch_sigma(sigma: torch.Tensor, batch_size: int, num_assets: int) -> torch.Tensor:
    if sigma.ndim == 2:
        return sigma.unsqueeze(0).expand(batch_size, num_assets, num_assets)
    if sigma.ndim == 3 and sigma.shape[0] == batch_size:
        return sigma
    raise ValueError("sigma must have shape (N, N) or (B, N, N)")


def compute_kkt_state(
    mu_hat: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    upper_bounds: torch.Tensor,
    problem: MinimalPortfolioProblem,
    lower_bounds: torch.Tensor = None,
    active_tolerance: float = 1e-5,
    compute_jacobian: bool = True,
    w_prev: torch.Tensor = None,
    turnover_penalty: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Extract primal-dual state and local sensitivity for the minimal QP.

    For the QP

    ``min 0.5*w.T*Q*w - mu.T*w + 0.5*rho*||w-w_prev||^2``

    subject to budget and box constraints, the effective quadratic and linear
    terms are ``Q + rho*I`` and ``mu + rho*w_prev``.  Setting ``rho=0`` gives
    the risk-only problem.

    the active-set Jacobian is computed from the free-coordinate KKT system.
    If ``F`` is the current free set, its block is

    ``Q_FF^{-1} - Q_FF^{-1}1(1.T Q_FF^{-1}1)^{-1}1.T Q_FF^{-1}``.

    Active coordinates have zero local sensitivity.  The active-set decision
    is intentionally treated as a discrete state; dual pressure remains
    differentiable with respect to ``mu_hat`` and the preliminary weights.
    """

    if active_tolerance <= 0:
        raise ValueError("active_tolerance must be positive")
    turnover_penalty = float(turnover_penalty)
    if not math.isfinite(turnover_penalty) or turnover_penalty < 0:
        raise ValueError("turnover_penalty must be finite and non-negative")
    if turnover_penalty != 0.0 and w_prev is None:
        raise ValueError("w_prev is required when turnover_penalty is non-zero")
    problem.validate_mu(mu_hat)
    problem.validate_sigma(sigma)
    if weights.shape != mu_hat.shape:
        raise ValueError("weights and mu_hat must have the same shape")
    if upper_bounds.ndim == 1:
        upper = upper_bounds.unsqueeze(0).expand_as(weights)
    elif upper_bounds.shape == weights.shape:
        upper = upper_bounds
    else:
        raise ValueError("upper_bounds must have shape (N,) or (B, N)")
    if lower_bounds is None:
        lower_bounds = torch.as_tensor(
            problem.lower_bounds, dtype=weights.dtype, device=weights.device
        )
    if lower_bounds.ndim == 1:
        lower = lower_bounds.unsqueeze(0).expand_as(weights)
    elif lower_bounds.shape == weights.shape:
        lower = lower_bounds
    else:
        raise ValueError("lower_bounds must have shape (N,) or (B, N)")

    batch_size, num_assets = weights.shape
    sigma_batch = _batch_sigma(sigma, batch_size, num_assets)
    eye = torch.eye(num_assets, dtype=sigma.dtype, device=sigma.device)
    q = 0.5 * (sigma_batch + sigma_batch.transpose(-1, -2))
    q = q + problem.eta * eye.unsqueeze(0)
    marginal_risk = torch.bmm(q, weights.unsqueeze(-1)).squeeze(-1)
    if w_prev is not None:
        if w_prev.shape != weights.shape:
            raise ValueError("w_prev and weights must have the same shape")
        if not torch.isfinite(w_prev).all():
            raise ValueError("w_prev contains NaN or infinite values")
    if turnover_penalty != 0.0:
        q = q + turnover_penalty * eye.unsqueeze(0)
        effective_mu = mu_hat + turnover_penalty * w_prev
    else:
        effective_mu = mu_hat
    quadratic_marginal = torch.bmm(q, weights.unsqueeze(-1)).squeeze(-1)
    gradient = quadratic_marginal - effective_mu

    active_lower = weights <= lower + active_tolerance
    active_upper = weights >= upper - active_tolerance
    free = ~(active_lower | active_upper)
    free_count = free.sum(dim=-1, keepdim=True).to(weights.dtype)
    free_nu = -(
        gradient * free.to(weights.dtype)
    ).sum(dim=-1, keepdim=True) / free_count.clamp_min(1.0)

    # If every coordinate is on a box bound, the budget multiplier is not
    # identified by a free-coordinate stationarity equation.  KKT signs give
    # an admissible interval instead:
    #
    #   lower-active i: gradient_i + nu >= 0  => nu >= -gradient_i
    #   upper-active i: gradient_i + nu <= 0  => nu <= -gradient_i.
    #
    # Pick the interval midpoint as a symmetric canonical representative.
    # When the budget itself pins every asset to only one side, the interval
    # is one-sided; its finite endpoint is a valid representative.  Assets
    # simultaneously classified at both bounds impose no restriction on nu.
    lower_only = active_lower & ~active_upper
    upper_only = active_upper & ~active_lower
    negative_gradient = -gradient
    positive_inf = torch.full_like(negative_gradient, torch.inf)
    negative_inf = torch.full_like(negative_gradient, -torch.inf)
    nu_lower = torch.where(
        lower_only, negative_gradient, negative_inf
    ).amax(dim=-1, keepdim=True)
    nu_upper = torch.where(
        upper_only, negative_gradient, positive_inf
    ).amin(dim=-1, keepdim=True)
    has_lower = lower_only.any(dim=-1, keepdim=True)
    has_upper = upper_only.any(dim=-1, keepdim=True)
    finite_nu_lower = torch.where(has_lower, nu_lower, torch.zeros_like(nu_lower))
    finite_nu_upper = torch.where(has_upper, nu_upper, torch.zeros_like(nu_upper))
    interval_midpoint = 0.5 * (finite_nu_lower + finite_nu_upper)
    no_free_nu = torch.where(
        has_lower & has_upper,
        interval_midpoint,
        torch.where(
            has_lower,
            nu_lower,
            torch.where(has_upper, nu_upper, -gradient.mean(dim=-1, keepdim=True)),
        ),
    )
    has_free = free_count > 0
    nu = torch.where(has_free, free_nu, no_free_nu)
    reduced_gradient = gradient + nu
    lower_dual = torch.relu(reduced_gradient) * active_lower.to(weights.dtype)
    upper_dual = torch.relu(-reduced_gradient) * active_upper.to(weights.dtype)
    pressure = lower_dual + upper_dual
    # With constraints written as lower-w <= 0 and w-upper <= 0, the full
    # stationarity residual is grad(f) + nu*1 - alpha + beta.  Keep it
    # distinct from the pre-box-dual reduced gradient above.
    kkt_stationarity_residual = reduced_gradient - lower_dual + upper_dual

    # The active set varies per sample, so the small KKT systems are solved in
    # a batch loop.  N is at most a few dozen in the current experiments and
    # this state is only constructed for the feedback pass.
    jacobian = None
    if compute_jacobian:
        jacobian = torch.zeros(
            batch_size,
            num_assets,
            num_assets,
            dtype=q.dtype,
            device=q.device,
        )
        for batch_index in range(batch_size):
            free_indices = torch.nonzero(free[batch_index], as_tuple=False).flatten()
            free_count_item = int(free_indices.numel())
            if free_count_item == 0:
                continue
            q_free = q[batch_index].index_select(0, free_indices).index_select(
                1, free_indices
            )
            ones = torch.ones(free_count_item, 1, dtype=q.dtype, device=q.device)
            kkt = torch.zeros(
                free_count_item + 1, free_count_item + 1,
                dtype=q.dtype, device=q.device,
            )
            kkt[:free_count_item, :free_count_item] = q_free
            kkt[:free_count_item, free_count_item:] = ones
            kkt[free_count_item:, :free_count_item] = ones.transpose(0, 1)
            rhs = torch.zeros(
                free_count_item + 1, free_count_item,
                dtype=q.dtype, device=q.device,
            )
            rhs[:free_count_item, :free_count_item] = torch.eye(
                free_count_item, dtype=q.dtype, device=q.device
            )
            solution = torch.linalg.solve(kkt, rhs)[:free_count_item]
            jacobian[batch_index][free_indices[:, None], free_indices[None, :]] = solution

    result = {
        "weights": weights,
        "active_lower": active_lower,
        "active_upper": active_upper,
        "free": free,
        "lower_dual": lower_dual,
        "upper_dual": upper_dual,
        "pressure": pressure,
        "budget_dual": nu.squeeze(-1),
        "budget_dual_has_free": has_free.squeeze(-1),
        "reduced_gradient": reduced_gradient,
        "kkt_stationarity_residual": kkt_stationarity_residual,
        "kkt_stationarity_residual_inf": (
            kkt_stationarity_residual.abs().amax(dim=-1)
        ),
        "marginal_risk": marginal_risk,
        "quadratic_marginal": quadratic_marginal,
    }
    if jacobian is not None:
        result["jacobian"] = jacobian
    return result
