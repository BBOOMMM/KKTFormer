"""Decision-focused losses for KKTFormer stage 5."""

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
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Evaluate the realized portfolio objective for one decision.

    The return target is the same ``H``-period aggregation used by the
    optimizer oracle.  The risk term is exactly the stage-1 quadratic
    objective, and transaction cost is an optional realized penalty.  The
    optimizer layer in stage 3 does not yet optimize turnover internally, so
    transaction cost is set to zero by default until the sequential state is
    implemented.

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
    objective_without_cost = problem.objective(weights, realized_mu, sigma)
    return_loss = -(weights * realized_mu).sum(dim=-1)
    risk_loss = objective_without_cost - return_loss

    batch_size = int(weights.numel() // problem.num_assets)
    flat_weights = weights.reshape(batch_size, problem.num_assets)
    if w_prev is None:
        turnover = torch.zeros(batch_size, dtype=weights.dtype, device=weights.device)
    else:
        if w_prev.shape != weights.shape:
            raise ValueError("w_prev must have the same shape as weights")
        turnover = (weights - w_prev).abs().sum(dim=-1).reshape(batch_size)
    cost_rate = _broadcast_batch_vector(
        transaction_cost_rate,
        batch_size=batch_size,
        dtype=weights.dtype,
        device=weights.device,
        name="transaction_cost_rate",
    )
    cost = turnover * cost_rate
    total = objective_without_cost.reshape(batch_size) + cost
    components = {
        "return_loss": return_loss.reshape(batch_size),
        "risk_loss": risk_loss.reshape(batch_size),
        "turnover": turnover,
        "transaction_cost": cost,
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
    )
    with torch.no_grad():
        oracle_loss, oracle_components = portfolio_objective_loss(
            oracle_weights,
            future_returns,
            sigma,
            problem,
            w_prev=w_prev,
            transaction_cost_rate=transaction_cost_rate,
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
    if weights.shape[:-1] != future_returns.shape[:-2]:
        raise ValueError("weights and future_returns have incompatible batch shapes")

    portfolio_returns = torch.einsum("...hn,...n->...h", future_returns, weights)
    portfolio_losses = -portfolio_returns
    var = torch.quantile(
        portfolio_losses.detach(), alpha, dim=-1, keepdim=True
    )
    smooth_excess = smooth_temperature * F.softplus(
        (portfolio_losses - var) / smooth_temperature
    )
    cvar = var.squeeze(-1) + smooth_excess.mean(dim=-1) / (1.0 - alpha)

    batch_size = int(weights.numel() // weights.shape[-1])
    if w_prev is None:
        turnover = torch.zeros(batch_size, dtype=weights.dtype, device=weights.device)
    else:
        if w_prev.shape != weights.shape:
            raise ValueError("w_prev must have the same shape as weights")
        turnover = (weights - w_prev).abs().sum(dim=-1).reshape(batch_size)
    cost_rate = _broadcast_batch_vector(
        transaction_cost_rate,
        batch_size=batch_size,
        dtype=weights.dtype,
        device=weights.device,
        name="transaction_cost_rate",
    )
    transaction_cost = turnover * cost_rate
    total = cvar.reshape(batch_size) + transaction_cost
    components = {
        "var": var.reshape(batch_size),
        "cvar": cvar.reshape(batch_size),
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "total_loss": total,
    }
    return total, components
