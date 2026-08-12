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


def portfolio_cvar_loss(
    weights: torch.Tensor,
    future_returns: torch.Tensor,
    alpha: float = 0.95,
    smooth_temperature: float = 1e-3,
    w_prev: Optional[torch.Tensor] = None,
    transaction_cost_rate: Optional[torch.Tensor] = None,
    turnover_penalty: float = 0.0,
    transaction_cost_smoothing: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a differentiable CVaR loss over the future return path.

    The empirical VaR threshold is detached from the model graph, while the
    tail excess uses a smooth softplus approximation.  This avoids a hard
    quantile branch in the training gradient and keeps the loss usable for the
    short ``H``-day paths used by KKTFormer-v0.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if smooth_temperature <= 0:
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
    var = torch.quantile(
        portfolio_losses.detach(), alpha, dim=-1, keepdim=True
    )
    smooth_excess = smooth_temperature * F.softplus(
        (portfolio_losses - var) / smooth_temperature
    )
    cvar = var.squeeze(-1) + smooth_excess.mean(dim=-1) / (1.0 - alpha)

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
