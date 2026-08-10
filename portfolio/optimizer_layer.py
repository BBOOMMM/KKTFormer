"""Differentiable long-only quadratic portfolio optimizer.

This is the stage-3 optimizer layer.  It solves the v0 problem from
``portfolio.problem`` with a fixed number of projected-gradient iterations.
The projection is onto the capped simplex

    {w | sum(w) = 1, 0 <= w_i <= upper_i}.

Both the iterations and the projection are implemented with PyTorch tensor
operations, so gradients can flow from the portfolio decision back to
``mu_hat``.  The layer intentionally does not expose KKT multipliers or a
decision Jacobian yet; those belong to later stages.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from portfolio.problem import MinimalPortfolioProblem


class _CappedSimplexProjection(torch.autograd.Function):
    """Capped-simplex projection with an active-set Jacobian.

    Differentiating through every branch of the bisection search gives a poor
    approximation to the derivative of the projection.  The projection is
    piecewise affine, however, so its exact local derivative is available once
    the active set is known.  For free coordinates F:

        dw_i / dv_j = 1[i=j] - 1/|F|,  i,j in F.

    Coordinates at zero or at their upper bound receive zero local gradient.
    This is the standard active-set subgradient used away from an active-set
    transition.
    """

    @staticmethod
    def forward(ctx, values, upper, target, bisection_steps):
        with torch.no_grad():
            target_tensor = torch.as_tensor(
                target, dtype=values.dtype, device=values.device
            )
            tau_low = (values - upper).amin(dim=-1)
            tau_high = values.amax(dim=-1)
            for _ in range(int(bisection_steps)):
                tau_mid = 0.5 * (tau_low + tau_high)
                projected = torch.clamp(
                    values - tau_mid.unsqueeze(-1), min=0.0
                )
                projected = torch.minimum(projected, upper)
                mass_too_high = projected.sum(dim=-1) > target_tensor
                tau_low = torch.where(mass_too_high, tau_mid, tau_low)
                tau_high = torch.where(mass_too_high, tau_high, tau_mid)

            tau = 0.5 * (tau_low + tau_high)
            projected = torch.clamp(values - tau.unsqueeze(-1), min=0.0)
            projected = torch.minimum(projected, upper)
            tolerance = 1e-7 if values.dtype == torch.float64 else 1e-5
            free = (projected > tolerance) & (projected < upper - tolerance)

            # Float32 bisection eventually stops changing tau.  Redistribute
            # the tiny remaining mass error over the free set so that the
            # budget constraint remains accurate in the dtype used by the
            # training model.
            residual = target_tensor - projected.sum(dim=-1, keepdim=True)
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
    target: float = 1.0,
    bisection_steps: int = 64,
) -> torch.Tensor:
    """Project vectors onto a simplex with per-asset upper bounds.

    The projection is the Euclidean projection:

    ``argmin_w 0.5 * ||w - values||^2``

    subject to ``sum(w) = target`` and ``0 <= w <= upper_bounds``.  It has the
    form ``clamp(values - tau, 0, upper_bounds)``; ``tau`` is found by a fixed
    number of bisection steps.  Fixed iterations keep the operation compatible
    with autograd and avoid a data-dependent Python stopping condition.
    """

    if not isinstance(values, torch.Tensor) or not isinstance(upper_bounds, torch.Tensor):
        raise TypeError("values and upper_bounds must be torch.Tensor objects")
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if upper_bounds.ndim not in (1, values.ndim):
        raise ValueError(
            "upper_bounds must have shape (N,) or the same number of dimensions "
            "as values"
        )
    if upper_bounds.shape[-1] != values.shape[-1]:
        raise ValueError("upper_bounds and values must have the same asset dimension")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")
    if not torch.isfinite(values).all():
        raise ValueError("values contains NaN or infinite values")
    if not torch.isfinite(upper_bounds).all():
        raise ValueError("upper_bounds contains NaN or infinite values")
    if (upper_bounds <= 0).any():
        raise ValueError("upper_bounds must be positive")

    if upper_bounds.ndim == 1:
        upper = upper_bounds.expand_as(values)
    else:
        if upper_bounds.shape != values.shape:
            raise ValueError("batched upper_bounds must match values shape")
        upper = upper_bounds

    target_tensor = torch.as_tensor(target, dtype=values.dtype, device=values.device)
    if (upper.sum(dim=-1) < target_tensor).any():
        raise ValueError("upper_bounds do not permit a feasible target sum")

    # At tau=amin(values-upper), every clipped coordinate is at its upper
    # bound, so the mass is at least target.  At tau=amax(values), the mass is
    # zero.  The mass is monotone decreasing in tau.  The custom Function
    # computes this projection in forward and uses its active-set Jacobian in
    # backward, rather than differentiating through the bisection branches.
    return _CappedSimplexProjection.apply(
        values, upper, float(target), int(bisection_steps)
    )


class DifferentiablePortfolioOptimizer(nn.Module):
    """Unrolled projected-gradient solver for the stage-3 QP.

    Args:
        problem: The :class:`MinimalPortfolioProblem` specification.
        num_iterations: Number of differentiable projected-gradient steps.
        bisection_steps: Number of steps for each capped-simplex projection.
        step_size: Optional fixed gradient step.  If omitted, a conservative
            ``0.95 / lambda_max(Q)`` is computed for every sample.
    """

    def __init__(
        self,
        problem: MinimalPortfolioProblem,
        num_iterations: int = 200,
        bisection_steps: int = 64,
        step_size: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if bisection_steps <= 0:
            raise ValueError("bisection_steps must be positive")
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive")
        self.problem = problem
        self.num_iterations = int(num_iterations)
        self.bisection_steps = int(bisection_steps)
        self.step_size = None if step_size is None else float(step_size)

    def _flatten_batch(
        self,
        mu_hat: torch.Tensor,
        sigma: torch.Tensor,
        upper_bounds: Optional[torch.Tensor],
        initial_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Tuple[int, ...]]:
        """Normalize supported input shapes to ``(batch, N)`` and ``(batch, N, N)``."""

        self.problem.validate_mu(mu_hat)
        self.problem.validate_sigma(sigma)
        original_batch_shape = tuple(mu_hat.shape[:-1])
        batch_size = int(mu_hat.numel() // self.problem.num_assets)
        mu = mu_hat.reshape(batch_size, self.problem.num_assets)

        if sigma.ndim == 2:
            sigma_batch = sigma.unsqueeze(0).expand(
                batch_size, self.problem.num_assets, self.problem.num_assets
            )
        elif tuple(sigma.shape[:-2]) == original_batch_shape:
            sigma_batch = sigma.reshape(
                batch_size, self.problem.num_assets, self.problem.num_assets
            )
        elif sigma.ndim == 3 and sigma.shape[0] == batch_size:
            sigma_batch = sigma
        else:
            raise ValueError(
                "sigma must have shape (N, N), (B, N, N), or "
                "the same leading batch shape as mu_hat"
            )

        def flatten_asset_tensor(
            tensor: torch.Tensor,
            name: str,
        ) -> torch.Tensor:
            if tensor.ndim == 1:
                if tensor.shape[0] != self.problem.num_assets:
                    raise ValueError(f"{name} has the wrong asset dimension")
                return tensor.unsqueeze(0).expand(batch_size, -1)
            if tuple(tensor.shape[:-1]) == original_batch_shape:
                return tensor.reshape(batch_size, self.problem.num_assets)
            if tensor.ndim == 2 and tensor.shape == (batch_size, self.problem.num_assets):
                return tensor
            raise ValueError(
                f"{name} must have shape (N,), (B, N), or the same leading "
                "batch shape as mu_hat"
            )

        if upper_bounds is None:
            upper = torch.as_tensor(
                self.problem.upper_bounds,
                dtype=mu.dtype,
                device=mu.device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            upper = flatten_asset_tensor(upper_bounds, "upper_bounds")
            upper = upper.to(dtype=mu.dtype, device=mu.device)

        if initial_weights is None:
            initial = None
        else:
            initial = flatten_asset_tensor(initial_weights, "initial_weights")
            initial = initial.to(dtype=mu.dtype, device=mu.device)

        return mu, sigma_batch, upper, initial, original_batch_shape

    def _step_size_for(self, q: torch.Tensor) -> torch.Tensor:
        if self.step_size is not None:
            smallest_eigenvalue = torch.linalg.eigvalsh(q)[..., 0]
            if (smallest_eigenvalue <= 0).any():
                raise ValueError("Sigma + eta I must be positive definite")
            return torch.full(
                (q.shape[0],),
                self.step_size,
                dtype=q.dtype,
                device=q.device,
            )
        eigenvalues = torch.linalg.eigvalsh(q)
        if (eigenvalues[..., 0] <= 0).any():
            raise ValueError("Sigma + eta I must be positive definite")
        largest_eigenvalue = eigenvalues[..., -1]
        return 0.95 / largest_eigenvalue

    def forward(
        self,
        mu_hat: torch.Tensor,
        sigma: torch.Tensor,
        factor_exposure: Optional[torch.Tensor] = None,
        w_prev: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
        initial_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Solve the stage-3 constrained portfolio problem.

        ``factor_exposure`` and ``w_prev`` are accepted for forward
        compatibility with the later constrained optimizer API.  They are not
        active in stage 3 because factor and turnover constraints are not yet
        part of the v0 problem.
        """

        del factor_exposure, w_prev
        mu, sigma_batch, upper, initial, original_batch_shape = self._flatten_batch(
            mu_hat, sigma, upper_bounds, initial_weights
        )
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma_batch).all():
            raise ValueError("mu_hat and sigma must be finite")

        n_assets = self.problem.num_assets
        eye = torch.eye(n_assets, dtype=sigma_batch.dtype, device=sigma_batch.device)
        q = 0.5 * (sigma_batch + sigma_batch.transpose(-1, -2))
        q = q + self.problem.eta * eye.unsqueeze(0)
        step = self._step_size_for(q)

        if initial is None:
            weights = project_capped_simplex(
                torch.zeros_like(mu), upper, bisection_steps=self.bisection_steps
            )
        else:
            weights = project_capped_simplex(
                initial, upper, bisection_steps=self.bisection_steps
            )

        last_step_norm = torch.zeros(
            mu.shape[0], dtype=mu.dtype, device=mu.device
        )
        for _ in range(self.num_iterations):
            gradient = torch.bmm(q, weights.unsqueeze(-1)).squeeze(-1) - mu
            candidate = weights - step.unsqueeze(-1) * gradient
            next_weights = project_capped_simplex(
                candidate, upper, bisection_steps=self.bisection_steps
            )
            last_step_norm = (next_weights - weights).abs().amax(dim=-1)
            weights = next_weights

        objective = self.problem.objective(weights, mu, sigma_batch)
        residuals = self.problem.constraint_residuals(weights)
        output_shape = original_batch_shape + (n_assets,)
        state: Dict[str, torch.Tensor] = {
            "objective": objective.reshape(original_batch_shape),
            "iterations": torch.as_tensor(
                self.num_iterations, dtype=torch.int64, device=mu.device
            ),
            "last_step_norm": last_step_norm.reshape(original_batch_shape),
            "budget_residual": residuals["budget"].reshape(original_batch_shape),
            "lower_violation": residuals["lower_violation"].reshape(output_shape),
            "upper_violation": residuals["upper_violation"].reshape(output_shape),
        }
        return weights.reshape(output_shape), state
