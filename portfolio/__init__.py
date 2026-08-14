"""Portfolio problem components used by KKTFormer."""

from .optimizer_layer import (
    DifferentiablePortfolioOptimizer,
    RiskBudgetedAllocator,
    project_bounded_simplex,
    project_capped_simplex,
    project_l1_ball,
)
from .losses import (
    decision_regret_loss,
    kkt_tail_ranking_loss,
    portfolio_cvar_loss,
    portfolio_risk_budget_loss,
    portfolio_objective_loss,
    sequence_decision_regret_loss,
)
from .kkt_feedback import compute_kkt_state
from .problem import MinimalPortfolioProblem

__all__ = [
    "DifferentiablePortfolioOptimizer",
    "RiskBudgetedAllocator",
    "MinimalPortfolioProblem",
    "decision_regret_loss",
    "kkt_tail_ranking_loss",
    "compute_kkt_state",
    "portfolio_cvar_loss",
    "portfolio_risk_budget_loss",
    "portfolio_objective_loss",
    "sequence_decision_regret_loss",
    "project_capped_simplex",
    "project_bounded_simplex",
    "project_l1_ball",
]
