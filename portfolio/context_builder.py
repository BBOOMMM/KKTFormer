"""Build point-in-time portfolio context from price data.

The context builder is the stage-2 data boundary for KKTFormer.  It computes
only quantities that are independent of the neural-network parameters:

* a historical log-return market window;
* a shrinkage covariance matrix ``Sigma_t``;
* price-derived factor exposures ``B_t``;
* the future simple-return path used only as a training/evaluation target;
* static constraint and transaction-cost parameters.

For a decision index ``t`` the historical price window is
``prices[t-lookback_window : t+1]``.  The first future return is
``prices[t+1] / prices[t] - 1``.  Thus the context never uses a price after
``t`` to construct ``Sigma_t`` or ``B_t``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from portfolio.problem import MinimalPortfolioProblem


DEFAULT_FACTOR_NAMES = ("market_beta", "momentum", "volatility")


@dataclass(frozen=True)
class PortfolioContextConfig:
    """Configuration for point-in-time context construction."""

    problem: MinimalPortfolioProblem
    covariance_epsilon: float = 1e-6
    transaction_cost_bps: float = 0.0
    factor_names: Tuple[str, ...] = DEFAULT_FACTOR_NAMES
    cross_sectional_zscore: bool = True

    def __post_init__(self) -> None:
        if self.covariance_epsilon <= 0:
            raise ValueError("covariance_epsilon must be positive")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if tuple(self.factor_names) != DEFAULT_FACTOR_NAMES:
            raise ValueError(
                "stage-2 price-only exposures must be exactly "
                f"{DEFAULT_FACTOR_NAMES}"
            )

    @property
    def transaction_cost_rate(self) -> float:
        """Convert basis points to a decimal cost rate."""

        return float(self.transaction_cost_bps) * 1e-4


def _as_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a price DataFrame."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas.DataFrame")
    if prices.shape[1] == 0:
        raise ValueError("prices must contain at least one asset column")

    frame = prices.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index)
        except (TypeError, ValueError) as exc:
            raise ValueError("prices index must be convertible to datetimes") from exc
    if frame.index.has_duplicates:
        raise ValueError("prices index must not contain duplicate dates")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError("prices contains NaN or infinite values")
    if (frame.to_numpy(dtype=np.float64) <= 0).any():
        raise ValueError("all prices must be strictly positive")
    return frame.astype(np.float64)


def _log_returns(price_window: np.ndarray) -> np.ndarray:
    """Compute log returns from a ``(L+1, N)`` price window."""

    if price_window.ndim != 2 or price_window.shape[0] < 2:
        raise ValueError("price_window must have shape (at least 2, num_assets)")
    return np.log(price_window[1:] / price_window[:-1])


def _cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score one asset exposure vector without producing NaNs."""

    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def estimate_covariance(
    price_window: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Estimate a positive-definite covariance matrix from past prices.

    Ledoit-Wolf shrinkage is used when scikit-learn is available.  The NumPy
    sample covariance is a fallback so context generation remains usable in a
    lightweight environment.  A symmetric epsilon diagonal is always added.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    returns = _log_returns(np.asarray(price_window, dtype=np.float64))
    if returns.shape[0] < 2:
        raise ValueError("at least two historical returns are required")

    try:
        from sklearn.covariance import LedoitWolf

        covariance = LedoitWolf().fit(returns).covariance_
    except ImportError:
        covariance = np.cov(returns, rowvar=False, ddof=1)

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + float(epsilon) * np.eye(covariance.shape[0])

    # Make the promise of positive definiteness explicit even when numerical
    # round-off leaves a tiny negative eigenvalue after covariance estimation.
    min_eigenvalue = np.linalg.eigvalsh(covariance).min()
    if min_eigenvalue <= 0:
        covariance = covariance + (
            float(epsilon) - min_eigenvalue
        ) * np.eye(covariance.shape[0])
    return covariance.astype(np.float32)


def build_price_factor_exposure(
    price_window: np.ndarray,
    cross_sectional_zscore: bool = True,
) -> np.ndarray:
    """Build price-only exposures ``[market_beta, momentum, volatility]``.

    ``price_window`` must contain only prices known at the decision time and
    have shape ``(L+1, N)``.  These are deliberately price-derived proxies,
    not a claim that the data contains official Barra or industry factors.
    """

    prices = np.asarray(price_window, dtype=np.float64)
    returns = _log_returns(prices)
    if returns.shape[0] < 2:
        raise ValueError("at least two historical returns are required")

    market_return = returns.mean(axis=1)
    market_centered = market_return - market_return.mean()
    market_variance = np.mean(market_centered**2)
    asset_centered = returns - returns.mean(axis=0, keepdims=True)
    if market_variance < 1e-12:
        market_beta = np.zeros(prices.shape[1], dtype=np.float64)
    else:
        market_beta = np.mean(
            asset_centered * market_centered[:, None], axis=0
        ) / market_variance

    momentum = prices[-1] / prices[0] - 1.0
    volatility = returns.std(axis=0, ddof=1)
    exposures = np.column_stack((market_beta, momentum, volatility))

    if cross_sectional_zscore:
        exposures = np.column_stack(
            [_cross_sectional_zscore(exposures[:, k]) for k in range(exposures.shape[1])]
        )
    return exposures.astype(np.float32)


def _future_simple_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from consecutive future prices."""

    return (prices[1:] / prices[:-1] - 1.0).astype(np.float32)


def _equal_weight(num_assets: int) -> np.ndarray:
    return np.full(num_assets, 1.0 / num_assets, dtype=np.float32)


def _build_context_at_frame(
    frame: pd.DataFrame,
    decision_index: int,
    config: PortfolioContextConfig,
) -> Dict[str, np.ndarray]:
    """Build one point-in-time context at an integer DataFrame index.

    ``decision_index`` denotes the last observed price.  It is not a future
    return index.  The first target return starts at this price and ends at
    the next available price.
    """

    values = frame.to_numpy(dtype=np.float64)
    n_assets = values.shape[1]
    if n_assets != config.problem.num_assets:
        raise ValueError(
            f"price asset count {n_assets} does not match "
            f"problem.num_assets {config.problem.num_assets}"
        )

    W = config.problem.lookback_window
    H = config.problem.horizon
    if not isinstance(decision_index, (int, np.integer)):
        raise TypeError("decision_index must be an integer")
    decision_index = int(decision_index)
    if decision_index < W:
        raise IndexError("not enough history before decision_index")
    if decision_index + H >= len(frame):
        raise IndexError("not enough future prices after decision_index")

    historical_prices = values[decision_index - W : decision_index + 1]
    future_prices = values[decision_index : decision_index + H + 1]
    historical_log_returns = _log_returns(historical_prices).astype(np.float32)

    return {
        "market_window": historical_log_returns[..., None],  # (W, N, 1)
        "future_returns": _future_simple_returns(future_prices),  # (H, N)
        "Sigma": estimate_covariance(
            historical_prices, epsilon=config.covariance_epsilon
        ),
        "factor_exposure": build_price_factor_exposure(
            historical_prices,
            cross_sectional_zscore=config.cross_sectional_zscore,
        ),
        "w_prev": _equal_weight(n_assets),
        "decision_date": np.datetime64(frame.index[decision_index], "ns"),
        "future_dates": frame.index[decision_index + 1 : decision_index + H + 1]
        .to_numpy(dtype="datetime64[ns]"),
        "lower_bounds": np.zeros(n_assets, dtype=np.float32),
        "upper_bounds": np.asarray(config.problem.upper_bounds, dtype=np.float32),
        "budget_target": np.float32(1.0),
        "transaction_cost_rate": np.float32(config.transaction_cost_rate),
        "factor_names": np.asarray(config.factor_names),
    }


def build_context_at(
    prices: pd.DataFrame,
    decision_index: int,
    config: PortfolioContextConfig,
) -> Dict[str, np.ndarray]:
    """Build one point-in-time context at an integer DataFrame index.

    ``decision_index`` denotes the last observed price.  It is not a future
    return index.  The first target return starts at this price and ends at
    the next available price.
    """

    return _build_context_at_frame(
        _as_price_frame(prices), decision_index=decision_index, config=config
    )


def _select_decision_indices(
    frame: pd.DataFrame,
    config: PortfolioContextConfig,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> np.ndarray:
    """Select regularly spaced, valid decision indices in a date range."""

    W = config.problem.lookback_window
    H = config.problem.horizon
    first = W
    last_exclusive = len(frame) - H
    if first >= last_exclusive:
        raise ValueError("price data is shorter than lookback_window + horizon")

    dates = frame.index
    start = pd.Timestamp(start_date) if start_date is not None else dates[first]
    end = pd.Timestamp(end_date) if end_date is not None else dates[last_exclusive - 1]
    candidates = np.arange(first, last_exclusive, dtype=np.int64)
    candidates = candidates[(dates[candidates] >= start) & (dates[candidates] <= end)]
    if len(candidates) == 0:
        raise ValueError("no valid decision dates in the requested range")

    # Rebalance frequency is measured in observations, not calendar days.
    return candidates[:: config.problem.rebalance_frequency]


def build_contexts(
    prices: pd.DataFrame,
    config: PortfolioContextConfig,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Dict[str, np.ndarray]:
    """Build and stack contexts for one train/validation/test split."""

    frame = _as_price_frame(prices)
    indices = _select_decision_indices(frame, config, start_date, end_date)
    # The public builder validates and copies a DataFrame for safety.  Avoid
    # repeating that work for every sample in a split; all samples use this
    # already validated frame.
    samples = [
        _build_context_at_frame(frame, int(index), config) for index in indices
    ]

    dynamic_keys = {
        "market_window",
        "future_returns",
        "Sigma",
        "factor_exposure",
        "w_prev",
        "decision_date",
        "future_dates",
    }
    context: Dict[str, np.ndarray] = {
        key: np.stack([sample[key] for sample in samples], axis=0)
        for key in dynamic_keys
    }
    context["decision_index"] = indices.astype(np.int64)
    context["lower_bounds"] = samples[0]["lower_bounds"]
    context["upper_bounds"] = samples[0]["upper_bounds"]
    context["budget_target"] = np.asarray(samples[0]["budget_target"], dtype=np.float32)
    context["transaction_cost_rate"] = np.asarray(
        samples[0]["transaction_cost_rate"], dtype=np.float32
    )
    context["factor_names"] = samples[0]["factor_names"]
    return context


def save_context_cache(context: Dict[str, np.ndarray], output_path: Union[str, Path]) -> None:
    """Save one split's contexts as a compressed NumPy archive."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **context)


def load_context_cache(input_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load a context archive created by :func:`save_context_cache`."""

    with np.load(input_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_context_cache_from_csv(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: PortfolioContextConfig,
    data_pool: Optional[int] = None,
    split_ranges: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, Path]:
    """Build train/val/test archives from a Date-indexed price CSV."""

    frame = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    if data_pool is not None:
        if data_pool <= 0 or data_pool > frame.shape[1]:
            raise ValueError("data_pool must be between 1 and the number of assets")
        frame = frame.iloc[:, :data_pool]

    if frame.shape[1] != config.problem.num_assets:
        raise ValueError(
            f"CSV has {frame.shape[1]} selected assets but problem expects "
            f"{config.problem.num_assets}"
        )

    if split_ranges is None:
        split_ranges = {
            "train": ("2000-01-01", "2016-12-31"),
            "val": ("2017-01-01", "2019-12-31"),
            "test": ("2020-01-01", "2024-12-31"),
        }

    output_dir = Path(output_dir)
    output_paths: Dict[str, Path] = {}
    for split, (start_date, end_date) in split_ranges.items():
        context = build_contexts(frame, config, start_date, end_date)
        path = output_dir / f"{split}.npz"
        save_context_cache(context, path)
        output_paths[split] = path
    return output_paths
