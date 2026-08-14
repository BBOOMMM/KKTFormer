"""Decision-focused losses for KKTFormer stage 5."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from portfolio.problem import MinimalPortfolioProblem


def _broadcast_batch_vector(
    value: Optional[torch.Tensor],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, dtype=dtype, device=device)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=dtype, device=device)
    else:
        value = value.to(dtype=dtype, device=device)
    if value.ndim == 0:
        return value.expand(batch_size)
    if value.ndim == 1 and value.shape[0] == batch_size:
        return value
    raise ValueError(f"{name} must be scalar or have shape (batch_size,)")


def portfolio_objective_loss(
    weights: torch.Tensor,
    future_returns: torch.Tensor,
    sigma: torch.Tensor,
    problem: MinimalPortfolioProblem,
    w_prev: Optional[torch.Tensor] = None,
    transaction_cost_rate: Optional[torch.Tensor] = None,
    turnover_penalty: float = 0.0,
    transaction_cost_smoothing: float = 1e-4,
    entropy_regularization: float = 0.0,
    entropy_epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Evaluate the realized portfolio objective for one decision.

    The return target is the same ``H``-period aggregation used by the
    optimizer oracle.  The risk term is exactly the stage-1 quadratic
    objective.  When supplied, quadratic turnover and smooth transaction cost
    use the same convention as the stage-7 optimizer; the reported
    ``transaction_cost`` component remains the actual L1 cost.

    Returns:
        loss: Tensor with shape ``(B,)``.
        components: return, risk, transaction-cost and total-loss tensors.
    """

    if weights.ndim < 1:
        raise ValueError("weights must have shape (..., N)")
    if future_returns.ndim < 2:
        raise ValueError("future_returns must have shape (..., H, N)")
    if weights.shape[:-1] != future_returns.shape[:-2]:
        raise ValueError("weights and future_returns have incompatible batch shapes")

    realized_mu = problem.aggregate_future_returns(future_returns)
    objective_without_cost = problem.objective(
        weights,
        realized_mu,
        sigma,
        entropy_regularization=entropy_regularization,
        entropy_epsilon=entropy_epsilon,
    )
    return_loss = -(weights * realized_mu).sum(dim=-1)
    epsilon = float(entropy_epsilon)
    safe_weights = weights + epsilon
    entropy_penalty = float(entropy_regularization) * (
        safe_weights * torch.log(safe_weights) - epsilon * math.log(epsilon)
    ).sum(dim=-1)
    risk_loss = objective_without_cost - return_loss - entropy_penalty

    batch_size = int(weights.numel() // problem.num_assets)
    flat_weights = weights.reshape(batch_size, problem.num_assets)
    if w_prev is None:
        turnover = torch.zeros(batch_size, dtype=weights.dtype, device=weights.device)
        delta = torch.zeros_like(flat_weights)
    else:
        if w_prev.shape != weights.shape:
            raise ValueError("w_prev must have the same shape as weights")
        delta = (weights - w_prev).reshape(batch_size, problem.num_assets)
        turnover = delta.abs().sum(dim=-1)
    cost_rate = _broadcast_batch_vector(
        transaction_cost_rate,
        batch_size=batch_size,
        dtype=weights.dtype,
        device=weights.device,
        name="transaction_cost_rate",
    )
    if turnover_penalty < 0:
        raise ValueError("turnover_penalty cannot be negative")
    if transaction_cost_smoothing <= 0:
        raise ValueError("transaction_cost_smoothing must be positive")
    quadratic_turnover = 0.5 * float(turnover_penalty) * delta.square().sum(dim=-1)
    smooth_transaction_cost = cost_rate * (
        torch.sqrt(delta.square() + transaction_cost_smoothing**2)
        - transaction_cost_smoothing
    ).sum(dim=-1)
    cost = cost_rate * turnover
    total = objective_without_cost.reshape(batch_size) + quadratic_turnover + smooth_transaction_cost
    components = {
        "return_loss": return_loss.reshape(batch_size),
        "risk_loss": risk_loss.reshape(batch_size),
        "entropy": (
            -weights * torch.log(weights.clamp_min(float(entropy_epsilon)))
        ).sum(dim=-1).reshape(batch_size),
        "entropy_penalty": entropy_penalty.reshape(batch_size),
        "turnover": turnover,
        "transaction_cost": cost,
        "smooth_transaction_cost": smooth_transaction_cost,
        "turnover_penalty": quadratic_turnover,
        "total_loss": total,
    }
    return total, components


def decision_regret_loss(
    predicted_weights: torch.Tensor,
    oracle_weights: torch.Tensor,
    future_returns: torch.Tensor,
    sigma: torch.Tensor,
    problem: MinimalPortfolioProblem,
    w_prev: Optional[torch.Tensor] = None,
    transaction_cost_rate: Optional[torch.Tensor] = None,
    turnover_penalty: float = 0.0,
    transaction_cost_smoothing: float = 1e-4,
    entropy_regularization: float = 0.0,
    entropy_epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute predicted-vs-oracle realized portfolio objective regret.

    ``oracle_weights`` must be obtained with the same optimizer, covariance,
    constraints, previous weights and transaction-cost convention.  It is
    detached from the model graph by the caller; this function also detaches
    its objective contribution so the oracle cannot receive gradients.
    """

    predicted_loss, predicted_components = portfolio_objective_loss(
        predicted_weights,
        future_returns,
        sigma,
        problem,
        w_prev=w_prev,
        transaction_cost_rate=transaction_cost_rate,
        turnover_penalty=turnover_penalty,
        transaction_cost_smoothing=transaction_cost_smoothing,
        entropy_regularization=entropy_regularization,
        entropy_epsilon=entropy_epsilon,
    )
    with torch.no_grad():
        oracle_loss, oracle_components = portfolio_objective_loss(
            oracle_weights,
            future_returns,
            sigma,
            problem,
            w_prev=w_prev,
            transaction_cost_rate=transaction_cost_rate,
            turnover_penalty=turnover_penalty,
            transaction_cost_smoothing=transaction_cost_smoothing,
            entropy_regularization=entropy_regularization,
            entropy_epsilon=entropy_epsilon,
        )
    regret = predicted_loss - oracle_loss.detach()
    components = {
        "predicted_objective": predicted_loss,
        "oracle_objective": oracle_loss.detach(),
        "regret": regret,
        "predicted_return_loss": predicted_components["return_loss"],
        "oracle_return_loss": oracle_components["return_loss"].detach(),
    }
    return regret, components


def sequence_decision_regret_loss(
    predicted_weights: torch.Tensor,
    oracle_weights: torch.Tensor,
    realized_returns: torch.Tensor,
    sigma: torch.Tensor,
    problem: MinimalPortfolioProblem,
    w_prev: Optional[torch.Tensor] = None,
    transaction_cost_rate: Optional[torch.Tensor] = None,
    turnover_penalty: float = 0.0,
    transaction_cost_smoothing: float = 1e-4,
    entropy_regularization: float = 0.0,
    entropy_epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute per-token realized decision regret for a portfolio sequence.

    Every ``(..., N)`` slice is one decision. ``realized_returns`` supplies
    the next-period asset returns known only to the training oracle. Both
    decisions are evaluated under the same quadratic risk, costs and previous
    position. The oracle contribution is detached from the model graph.
    """

    if predicted_weights.shape != realized_returns.shape:
        raise ValueError(
            "predicted_weights and realized_returns must have identical shape"
        )
    if oracle_weights.shape != predicted_weights.shape:
        raise ValueError(
            "oracle_weights and predicted_weights must have identical shape"
        )
    if sigma.shape != predicted_weights.shape[:-1] + (
        problem.num_assets,
        problem.num_assets,
    ):
        raise ValueError("sigma batch shape must match the decision sequence")
    if w_prev is not None and w_prev.shape != predicted_weights.shape:
        raise ValueError("w_prev must have the same shape as predicted_weights")

    predicted_objective = problem.objective(
        predicted_weights,
        realized_returns,
        sigma,
        w_prev=w_prev,
        turnover_penalty=turnover_penalty,
        transaction_cost_rate=transaction_cost_rate,
        transaction_cost_smoothing=transaction_cost_smoothing,
        entropy_regularization=entropy_regularization,
        entropy_epsilon=entropy_epsilon,
    )
    with torch.no_grad():
        oracle_objective = problem.objective(
            oracle_weights,
            realized_returns,
            sigma,
            w_prev=w_prev,
            turnover_penalty=turnover_penalty,
            transaction_cost_rate=transaction_cost_rate,
            transaction_cost_smoothing=transaction_cost_smoothing,
            entropy_regularization=entropy_regularization,
            entropy_epsilon=entropy_epsilon,
        )

    predicted_return_loss = -(
        predicted_weights * realized_returns
    ).sum(dim=-1)
    oracle_return_loss = -(oracle_weights * realized_returns).sum(dim=-1)
    regret = predicted_objective - oracle_objective.detach()
    return regret, {
        "predicted_objective": predicted_objective,
        "oracle_objective": oracle_objective.detach(),
        "regret": regret,
        "predicted_return_loss": predicted_return_loss,
        "oracle_return_loss": oracle_return_loss.detach(),
    }


def portfolio_cvar_loss(
    weights: torch.Tensor,
    future_returns: torch.Tensor,
    alpha: float = 0.95,
    variant: str = "sit",
    smooth_temperature: float = 1e-3,
    w_prev: Optional[torch.Tensor] = None,
    transaction_cost_rate: Optional[torch.Tensor] = None,
    turnover_penalty: float = 0.0,
    transaction_cost_smoothing: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute CVaR over the future return path.

    ``variant="sit"`` exactly matches the released SIT objective: VaR remains
    in the computation graph and the tail excess is a hard ReLU.  The
    detached-VaR softplus approximation is retained as ``variant="smooth"``
    for ablation experiments.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    variant = str(variant).lower()
    if variant not in {"sit", "smooth"}:
        raise ValueError("variant must be one of sit or smooth")
    if variant == "smooth" and smooth_temperature <= 0:
        raise ValueError("smooth_temperature must be positive")
    sequence_weights = weights.shape == future_returns.shape
    static_weights = weights.shape[:-1] == future_returns.shape[:-2]
    if not (sequence_weights or static_weights):
        raise ValueError("weights must be (...,N) or (...,H,N) matching future_returns")

    if sequence_weights:
        portfolio_returns = (future_returns * weights).sum(dim=-1)
    else:
        portfolio_returns = torch.einsum("...hn,...n->...h", future_returns, weights)
    portfolio_losses = -portfolio_returns
    if variant == "sit":
        var = torch.quantile(portfolio_losses, alpha, dim=-1, keepdim=True)
        excess = F.relu(portfolio_losses - var)
    else:
        var = torch.quantile(
            portfolio_losses.detach(), alpha, dim=-1, keepdim=True
        )
        excess = smooth_temperature * F.softplus(
            (portfolio_losses - var) / smooth_temperature
        )
    cvar = var.squeeze(-1) + excess.mean(dim=-1) / (1.0 - alpha)

    batch_size = int(future_returns.numel() // (future_returns.shape[-2] * future_returns.shape[-1]))
    if sequence_weights:
        if w_prev is None:
            first_delta = torch.zeros_like(weights[..., 0, :])
        elif w_prev.shape == weights.shape[:-2] + (weights.shape[-1],):
            first_delta = weights[..., 0, :] - w_prev
        elif w_prev.shape == weights.shape:
            first_delta = weights[..., 0, :] - w_prev[..., 0, :]
        else:
            raise ValueError("w_prev must have shape (...,N) or (...,H,N)")
        later_delta = weights[..., 1:, :] - weights[..., :-1, :]
        delta_path = torch.cat((first_delta.unsqueeze(-2), later_delta), dim=-2)
        turnover = delta_path.abs().sum(dim=(-1, -2)).reshape(batch_size)
        smooth_norm = (
            torch.sqrt(delta_path.square() + transaction_cost_smoothing**2)
            - transaction_cost_smoothing
        ).sum(dim=(-1, -2)).reshape(batch_size)
        quadratic_norm = delta_path.square().sum(dim=(-1, -2)).reshape(batch_size)
    else:
        if w_prev is None:
            delta = torch.zeros_like(weights).reshape(batch_size, weights.shape[-1])
        else:
            if w_prev.shape != weights.shape:
                raise ValueError("w_prev must have the same shape as weights")
            delta = (weights - w_prev).reshape(batch_size, weights.shape[-1])
        turnover = delta.abs().sum(dim=-1)
        smooth_norm = (
            torch.sqrt(delta.square() + transaction_cost_smoothing**2)
            - transaction_cost_smoothing
        ).sum(dim=-1)
        quadratic_norm = delta.square().sum(dim=-1)
    cost_rate = _broadcast_batch_vector(
        transaction_cost_rate,
        batch_size=batch_size,
        dtype=weights.dtype,
        device=weights.device,
        name="transaction_cost_rate",
    )
    if turnover_penalty < 0:
        raise ValueError("turnover_penalty cannot be negative")
    if transaction_cost_smoothing <= 0:
        raise ValueError("transaction_cost_smoothing must be positive")
    transaction_cost = turnover * cost_rate
    smooth_transaction_cost = cost_rate * smooth_norm
    quadratic_turnover = 0.5 * float(turnover_penalty) * quadratic_norm
    total = cvar.reshape(batch_size) + smooth_transaction_cost + quadratic_turnover
    components = {
        "var": var.reshape(batch_size),
        "cvar": cvar.reshape(batch_size),
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "smooth_transaction_cost": smooth_transaction_cost,
        "turnover_penalty": quadratic_turnover,
        "total_loss": total,
    }
    return total, components


def portfolio_risk_budget_loss(
    weights: torch.Tensor,
    future_returns: torch.Tensor,
    alpha: float = 0.95,
    downside_temperature: float = 1e-2,
    drawdown_temperature: float = 1e-2,
    downside_weight: float = 0.25,
    drawdown_weight: float = 0.10,
    w_prev: Optional[torch.Tensor] = None,
    turnover_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Smooth decision-focused risk-budget objective.

    This objective complements CVaR with two path-level terms that are useful
    for an actual portfolio: a soft downside semideviation and a smooth
    maximum drawdown surrogate.  The latter is computed on cumulative log
    wealth, so it is differentiable through every allocation in the horizon.
    A quadratic turnover term makes the same decision layer usable when a
    trading budget is supplied.  It is deliberately scale-normalized by the
    horizon and does not replace the hard constraints in the optimizer.
    """

    if weights.shape != future_returns.shape:
        raise ValueError("weights and future_returns must both have shape (..., H, N)")
    if downside_temperature <= 0 or drawdown_temperature <= 0:
        raise ValueError("risk smoothing temperatures must be positive")
    if min(downside_weight, drawdown_weight, turnover_weight) < 0:
        raise ValueError("risk objective weights cannot be negative")
    portfolio_returns = (weights * future_returns).sum(dim=-1)
    losses = -portfolio_returns
    var = torch.quantile(losses, alpha, dim=-1, keepdim=True)
    cvar = var.squeeze(-1) + torch.relu(losses - var).mean(dim=-1) / (1.0 - alpha)

    # ``softplus`` is a smooth positive-part operator and avoids the zero
    # gradient region of a hard downside mask during early training.
    downside = downside_temperature * torch.nn.functional.softplus(
        -portfolio_returns / downside_temperature
    ).mean(dim=-1)
    log_wealth = torch.log1p(portfolio_returns.clamp(min=-0.99)).cumsum(dim=-1)
    running_peak = torch.cummax(log_wealth, dim=-1).values
    drawdown = running_peak - log_wealth
    smooth_max_drawdown = drawdown_temperature * torch.logsumexp(
        drawdown / drawdown_temperature, dim=-1
    )

    if w_prev is None:
        turnover = torch.zeros_like(cvar)
    else:
        first = (weights[..., 0, :] - w_prev).square().sum(dim=-1)
        later = (
            (weights[..., 1:, :] - weights[..., :-1, :]).square().sum(dim=-1)
            if weights.shape[-2] > 1
            else torch.zeros_like(first).unsqueeze(-1)
        )
        turnover = (first + later.sum(dim=-1)) / float(weights.shape[-2])

    total = (
        cvar
        + float(downside_weight) * downside
        + float(drawdown_weight) * smooth_max_drawdown
        + float(turnover_weight) * turnover
    )
    return total, {
        "cvar": cvar,
        "var": var.squeeze(-1),
        "downside": downside,
        "smooth_max_drawdown": smooth_max_drawdown,
        "turnover": turnover,
        "total_loss": total,
    }


def kkt_tail_ranking_loss(
    allocation_logits: torch.Tensor,
    weights: torch.Tensor,
    future_returns: torch.Tensor,
    pressure: torch.Tensor,
    tail_alpha: float = 0.95,
    pressure_scale: float = 1.0,
    ranking_temperature: float = 1.0,
    pressure_clip: float = 5.0,
    pressure_epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Rank assets on CVaR-tail days with detached KKT pressure weights.

    The tail scenarios and pair labels are targets, not trainable quantities.
    Likewise, probe pressure only controls credit assignment and is detached so
    the model cannot reduce this loss by manipulating its sample weights.  Pair
    losses are averaged (rather than summed) to keep the scale comparable when
    the number of assets changes.

    Args:
        allocation_logits: Refined pre-softmax scores with shape ``(B,H,N)``.
        weights: Final portfolio weights with shape ``(B,H,N)``.
        future_returns: Realized next-period asset returns, shape ``(B,H,N)``.
        pressure: Probe ``lower_dual + upper_dual``, shape ``(B,H,N)``.
        tail_alpha: Quantile used only to select bad portfolio-return days.
        pressure_scale: Kappa in ``1 + kappa * (p_i + p_j) / 2``.
        ranking_temperature: Positive temperature for pairwise logit gaps.
        pressure_clip: Upper bound after mean normalization of pressure.
    """

    if allocation_logits.ndim != 3:
        raise ValueError("allocation_logits must have shape (B, H, N)")
    if weights.shape != allocation_logits.shape:
        raise ValueError("weights must have the same shape as allocation_logits")
    if future_returns.shape != allocation_logits.shape:
        raise ValueError("future_returns must have the same shape as allocation_logits")
    if pressure.shape != allocation_logits.shape:
        raise ValueError("pressure must have the same shape as allocation_logits")
    if allocation_logits.shape[-1] < 2:
        raise ValueError("KKT tail ranking requires at least two assets")
    if not 0.0 < tail_alpha < 1.0:
        raise ValueError("tail_alpha must be in (0, 1)")
    if pressure_scale < 0.0:
        raise ValueError("pressure_scale must be non-negative")
    if ranking_temperature <= 0.0:
        raise ValueError("ranking_temperature must be positive")
    if pressure_clip <= 0.0:
        raise ValueError("pressure_clip must be positive")
    if pressure_epsilon <= 0.0:
        raise ValueError("pressure_epsilon must be positive")

    # CVaR identifies the bad time steps, but KTR must not alter the original
    # CVaR graph through its discrete scenario selection.
    with torch.no_grad():
        portfolio_losses = -(weights.detach() * future_returns.detach()).sum(dim=-1)
        tail_var = torch.quantile(
            portfolio_losses, tail_alpha, dim=-1, keepdim=True
        )
        tail_mask = (portfolio_losses >= tail_var).to(allocation_logits.dtype)

        detached_pressure = pressure.detach().clamp_min(0.0)
        pressure_mean = detached_pressure.mean(dim=-1, keepdim=True)
        normalized_pressure = (
            detached_pressure / (pressure_mean + pressure_epsilon)
        ).clamp(max=pressure_clip)

        asset_count = allocation_logits.shape[-1]
        pair_i, pair_j = torch.triu_indices(
            asset_count,
            asset_count,
            offset=1,
            device=allocation_logits.device,
        )
        return_gap = future_returns.detach()[..., pair_i] - future_returns.detach()[
            ..., pair_j
        ]
        pair_label = torch.sign(return_gap)
        valid_pair = (pair_label != 0).to(allocation_logits.dtype)
        pair_weight = 1.0 + pressure_scale * 0.5 * (
            normalized_pressure[..., pair_i] + normalized_pressure[..., pair_j]
        )

    logit_gap = allocation_logits[..., pair_i] - allocation_logits[..., pair_j]
    pair_loss = F.softplus(
        -pair_label * logit_gap / ranking_temperature
    )
    valid_count = valid_pair.sum(dim=-1).clamp_min(1.0)
    ranking_by_time = (pair_weight * valid_pair * pair_loss).sum(dim=-1) / valid_count
    tail_count = tail_mask.sum(dim=-1).clamp_min(1.0)
    ranking_by_sample = (tail_mask * ranking_by_time).sum(dim=-1) / tail_count

    components = {
        "tail_var": tail_var.squeeze(-1),
        "tail_fraction": tail_mask.mean(dim=-1),
        "mean_pressure": detached_pressure.mean(dim=(-1, -2)),
        "nonzero_pressure_ratio": (detached_pressure > 0).to(
            allocation_logits.dtype
        ).mean(dim=(-1, -2)),
        "mean_pair_weight": pair_weight.mean(dim=(-1, -2)),
        "ranking_loss": ranking_by_sample,
    }
    return ranking_by_sample, components
