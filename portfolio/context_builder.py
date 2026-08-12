"""Build point-in-time portfolio context from price data.

The context builder is the stage-2 data boundary for KKTFormer.  It computes
only quantities that are independent of the neural-network parameters:

* ``H`` rolling log-return paths, one for each SIT horizon token;
* rolling shrinkage covariance matrices ``Sigma_{t+h}``;
* rolling price-derived factor exposures ``B_{t+h}``;
* the future simple-return path used only as a training/evaluation target;
* static constraint and transaction-cost parameters.

For a decision index ``t``, horizon token ``h`` contains the ``W`` prices
ending at ``t+h``.  Its execution date is ``t+h+1`` and its target is the
return beginning on that date.  This exactly matches SIT's rolling-path
layout.  Each token's covariance and factor exposures are constructed from the
same rolling price window as that token, so their decision timestamps match.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from portfolio.problem import MinimalPortfolioProblem
from utils.timefeatures import time_features


DEFAULT_FACTOR_NAMES = ("market_beta", "momentum", "volatility")


@dataclass(frozen=True)
class PortfolioContextConfig:
    """Configuration for point-in-time context construction."""

    problem: MinimalPortfolioProblem
    covariance_epsilon: float = 1e-6
    transaction_cost_bps: float = 0.0
    factor_names: Tuple[str, ...] = DEFAULT_FACTOR_NAMES
    cross_sectional_zscore: bool = True
    factor_lower: Optional[Union[float, Sequence[float]]] = None
    factor_upper: Optional[Union[float, Sequence[float]]] = None
    industry_exposure: Optional[np.ndarray] = None
    industry_names: Tuple[str, ...] = ()
    industry_lower: Optional[Union[float, Sequence[float]]] = None
    industry_upper: Optional[Union[float, Sequence[float]]] = None

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
        for name, bound in (
            ("factor_lower", self.factor_lower),
            ("factor_upper", self.factor_upper),
            ("industry_lower", self.industry_lower),
            ("industry_upper", self.industry_upper),
        ):
            if bound is not None and np.isnan(np.asarray(bound, dtype=np.float64)).any():
                raise ValueError(f"{name} cannot contain NaN")
        if self.industry_exposure is not None:
            exposure = np.asarray(self.industry_exposure, dtype=np.float64)
            if exposure.ndim != 2 or exposure.shape[0] != self.problem.num_assets:
                raise ValueError("industry_exposure must have shape (num_assets, num_industries)")
            if not np.isfinite(exposure).all():
                raise ValueError("industry_exposure must be finite")
            if exposure.shape[1] == 0:
                raise ValueError("industry_exposure must contain at least one industry")
            object.__setattr__(self, "industry_exposure", exposure.astype(np.float32))
            if self.industry_names and len(self.industry_names) != exposure.shape[1]:
                raise ValueError("industry_names must match industry_exposure columns")
            if not self.industry_names:
                object.__setattr__(
                    self,
                    "industry_names",
                    tuple(f"industry_{i}" for i in range(exposure.shape[1])),
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


def _normalise_bound_vector(
    value: Optional[Union[float, Sequence[float]]],
    size: int,
    name: str,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float32)
    elif array.ndim == 1 and array.shape[0] == size:
        array = array.astype(np.float32)
    else:
        raise ValueError(f"{name} must be scalar or have length {size}")
    if np.isnan(array).any():
        raise ValueError(f"{name} cannot contain NaN")
    return array


def load_industry_exposure(
    path: Union[str, Path], asset_names: Sequence[str]
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Load an industry one-hot matrix from ``.npy`` or a simple CSV.

    CSV mappings may contain ``asset`` and ``industry`` columns.  A numeric
    matrix with one row per asset is also accepted.
    """

    path = Path(path)
    asset_names = list(asset_names)
    if path.suffix.lower() == ".npy":
        exposure = np.load(path)
        if exposure.ndim != 2 or exposure.shape[0] != len(asset_names):
            raise ValueError("industry .npy must have shape (num_assets, num_industries)")
        return np.asarray(exposure, dtype=np.float32), tuple(
            f"industry_{i}" for i in range(exposure.shape[1])
        )

    mapping = pd.read_csv(path)
    lower_columns = {str(column).lower(): column for column in mapping.columns}
    asset_column = lower_columns.get("asset") or lower_columns.get("symbol")
    industry_column = lower_columns.get("industry") or lower_columns.get("sector")
    if asset_column is None or industry_column is None:
        numeric = mapping.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if numeric.shape[0] != len(asset_names) or numeric.ndim != 2 or numeric.shape[1] == 0:
            raise ValueError("industry CSV must have asset/industry columns or a numeric matrix")
        return numeric, tuple(str(column) for column in mapping.select_dtypes(include=[np.number]).columns)

    categories = sorted(mapping[industry_column].astype(str).unique().tolist())
    category_index = {category: index for index, category in enumerate(categories)}
    asset_index = {str(asset): index for index, asset in enumerate(asset_names)}
    exposure = np.zeros((len(asset_names), len(categories)), dtype=np.float32)
    for _, row in mapping.iterrows():
        asset = str(row[asset_column])
        if asset not in asset_index:
            raise ValueError(f"industry mapping contains unknown asset {asset!r}")
        exposure[asset_index[asset], category_index[str(row[industry_column])]] = 1.0
    if (exposure.sum(axis=1) == 0).any():
        raise ValueError("every asset must have an industry mapping")
    return exposure, tuple(categories)


def _build_context_at_frame(
    frame: pd.DataFrame,
    decision_index: int,
    config: PortfolioContextConfig,
    allow_incomplete_future: bool = False,
) -> Dict[str, np.ndarray]:
    """Build one point-in-time context at an integer DataFrame index.

    ``decision_index`` denotes the last observed price.  The first target
    return starts at the next price, so a context anchored at ``R-1`` has
    execution date ``R``.  When ``allow_incomplete_future`` is true, the
    target path is zero-padded for the final test decisions; the first target
    remains an actual observed return and is the only one used by the SIT
    compatible backtest.
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
    if decision_index < W - 1:
        raise IndexError("not enough history before decision_index")
    first_future_index = decision_index + 1
    if first_future_index + 1 >= len(frame):
        raise IndexError("not enough future prices after decision_index")

    future_end_exclusive = min(first_future_index + H + 1, len(frame))
    future_prices = values[first_future_index:future_end_exclusive]
    future_length = future_prices.shape[0] - 1
    if future_length < 1:
        raise IndexError("not enough future prices after decision_index")
    if future_length < H and not allow_incomplete_future:
        raise IndexError("not enough complete future horizon after decision_index")

    # Construct H rolling paths, matching SIT's per-horizon token layout.
    # Each path contains W prices known immediately before that token's
    # execution date.  The leading zero preserves the exact W-dimensional
    # representation while retaining W-1 genuine log returns.
    path_prices = []
    future_dates = frame.index[first_future_index:future_end_exclusive].to_numpy(
        dtype="datetime64[ns]"
    )
    if future_dates.shape[0] < H + 1:
        future_dates = np.concatenate(
            [future_dates, np.repeat(future_dates[-1:], H + 1 - future_dates.shape[0])]
        )
    for step in range(H):
        end = decision_index + step
        available_end = min(end, len(frame) - 1)
        start = available_end - W + 1
        if start < 0:
            raise IndexError("not enough history for rolling log-return path")
        path = values[start : available_end + 1]
        if path.shape[0] < W:
            path = np.concatenate(
                [path, np.repeat(path[-1:], W - path.shape[0], axis=0)], axis=0
            )
        path_prices.append(path)
    path_prices = np.stack(path_prices, axis=0)
    log_return_path = np.concatenate(
        [np.zeros((H, 1, n_assets), dtype=np.float32),
         np.log(path_prices[:, 1:] / path_prices[:, :-1]).astype(np.float32)],
        axis=1,
    ).transpose(0, 2, 1)  # (H, N, W)
    future_returns = _future_simple_returns(future_prices)
    if future_length < H:
        future_returns = np.concatenate(
            [future_returns, np.zeros((H - future_length, n_assets), dtype=np.float32)],
            axis=0,
        )

    factor_lower = _normalise_bound_vector(
        config.factor_lower, len(config.factor_names), "factor_lower"
    )
    factor_upper = _normalise_bound_vector(
        config.factor_upper, len(config.factor_names), "factor_upper"
    )

    # Risk and factor context must share the timestamp of each horizon token.
    # Computing these once at t and broadcasting would pair X_{t+h} with
    # Sigma_t/B_t instead of Sigma_{t+h}/B_{t+h}.
    sigma_sequence = np.stack(
        [
            estimate_covariance(window, epsilon=config.covariance_epsilon)
            for window in path_prices
        ],
        axis=0,
    )
    factor_sequence = np.stack(
        [
            build_price_factor_exposure(
                window,
                cross_sectional_zscore=config.cross_sectional_zscore,
            )
            for window in path_prices
        ],
        axis=0,
    )

    result = {
        "log_return_path": log_return_path,  # (H, N, W)
        "date_feats": time_features(pd.to_datetime(future_dates[:H]), freq="B").T.astype(np.float32),
        "future_returns": future_returns,  # (H, N), zero-padded only if allowed
        "Sigma": sigma_sequence,  # (H, N, N)
        "factor_exposure": factor_sequence,  # (H, N, K)
        "w_prev": _equal_weight(n_assets),
        "decision_date": np.datetime64(frame.index[decision_index], "ns"),
        "future_dates": future_dates[:H],
        "future_valid_length": np.int64(future_length),
        "lower_bounds": np.asarray(config.problem.lower_bounds, dtype=np.float32),
        "upper_bounds": np.asarray(config.problem.upper_bounds, dtype=np.float32),
        "budget_target": np.float32(config.problem.budget_target),
        "transaction_cost_rate": np.float32(config.transaction_cost_rate),
        "factor_names": np.asarray(config.factor_names),
    }
    if factor_lower is not None:
        result["factor_lower"] = factor_lower
    if factor_upper is not None:
        result["factor_upper"] = factor_upper
    if config.industry_exposure is not None:
        result["industry_exposure"] = np.asarray(config.industry_exposure, dtype=np.float32)
        result["industry_names"] = np.asarray(config.industry_names)
        industry_lower = _normalise_bound_vector(
            config.industry_lower,
            result["industry_exposure"].shape[1],
            "industry_lower",
        )
        industry_upper = _normalise_bound_vector(
            config.industry_upper,
            result["industry_exposure"].shape[1],
            "industry_upper",
        )
        if industry_lower is not None:
            result["industry_lower"] = industry_lower
        if industry_upper is not None:
            result["industry_upper"] = industry_upper
    return result


def build_context_at(
    prices: pd.DataFrame,
    decision_index: int,
    config: PortfolioContextConfig,
    allow_incomplete_future: bool = False,
) -> Dict[str, np.ndarray]:
    """Build one point-in-time context at an integer DataFrame index.

    ``decision_index`` denotes the last observed price.  The first target
    return starts on the next execution date.
    """

    return _build_context_at_frame(
        _as_price_frame(prices),
        decision_index=decision_index,
        config=config,
        allow_incomplete_future=allow_incomplete_future,
    )


def _select_decision_indices(
    frame: pd.DataFrame,
    config: PortfolioContextConfig,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    allow_incomplete_future: bool = False,
) -> np.ndarray:
    """Select regularly spaced decision indices in a date range.

    Date filtering is applied to the future execution path, not the decision
    date.  This is important for SIT compatibility: the decision at ``R-1``
    belongs to the prediction/return date ``R``.
    """

    W = config.problem.lookback_window
    H = config.problem.horizon
    first = W - 1
    last_exclusive = len(frame) - 2
    if not allow_incomplete_future:
        # H returns starting at t+1 require prices through t+H+1.
        last_exclusive = len(frame) - H - 1
    if first >= last_exclusive:
        raise ValueError("price data is shorter than lookback_window + horizon")

    dates = frame.index
    start = pd.Timestamp(start_date) if start_date is not None else dates[first]
    # ``end`` is the end of the observable split, not the last permissible
    # decision date.  The latter is determined below by requiring the entire
    # future horizon to remain inside the split.
    end = pd.Timestamp(end_date) if end_date is not None else dates[-1]
    candidates = np.arange(first, last_exclusive, dtype=np.int64)
    candidates = candidates[(dates[candidates + 1] >= start)]
    if allow_incomplete_future:
        candidates = candidates[dates[candidates + 1] <= end]
    else:
        candidates = candidates[
            (dates[candidates + H] <= end)
        ]
    if len(candidates) == 0:
        raise ValueError("no valid decision dates in the requested range")

    # Rebalance frequency is measured in observations, not calendar days.
    return candidates[:: config.problem.rebalance_frequency]


def _decision_indices_for_dates(
    frame: pd.DataFrame,
    execution_dates: Sequence[Union[str, pd.Timestamp]],
    config: PortfolioContextConfig,
    allow_incomplete_future: bool = False,
) -> np.ndarray:
    """Map execution dates ``R`` to KKT decision indices ``R-1``."""

    dates = frame.index
    indices = []
    for execution_date in execution_dates:
        execution_date = pd.Timestamp(execution_date)
        location = dates.get_indexer([execution_date])[0]
        if location < 0:
            raise ValueError(f"execution date {execution_date.date()} is not in the data")
        decision_index = int(location) - 1
        if decision_index < config.problem.lookback_window - 1:
            raise ValueError(
                f"execution date {execution_date.date()} has insufficient history"
            )
        if decision_index + 2 >= len(frame):
            raise ValueError(
                f"execution date {execution_date.date()} has no following return"
            )
        if not allow_incomplete_future and decision_index + config.problem.horizon + 1 >= len(frame):
            raise ValueError(
                f"execution date {execution_date.date()} has an incomplete future horizon"
            )
        indices.append(decision_index)
    if not indices:
        raise ValueError("execution_dates must contain at least one date")
    return np.asarray(indices, dtype=np.int64)


def build_contexts(
    prices: pd.DataFrame,
    config: PortfolioContextConfig,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    execution_dates: Optional[Sequence[Union[str, pd.Timestamp]]] = None,
    allow_incomplete_future: bool = False,
) -> Dict[str, np.ndarray]:
    """Build and stack contexts for one train/validation/test split."""

    frame = _as_price_frame(prices)
    if execution_dates is None:
        indices = _select_decision_indices(
            frame,
            config,
            start_date,
            end_date,
            allow_incomplete_future=allow_incomplete_future,
        )
    else:
        indices = _decision_indices_for_dates(
            frame,
            execution_dates,
            config,
            allow_incomplete_future=allow_incomplete_future,
        )
    # The public builder validates and copies a DataFrame for safety.  Avoid
    # repeating that work for every sample in a split; all samples use this
    # already validated frame.
    samples = [
        _build_context_at_frame(
            frame,
            int(index),
            config,
            allow_incomplete_future=allow_incomplete_future,
        )
        for index in indices
    ]

    dynamic_keys = {
        "log_return_path",
        "date_feats",
        "future_returns",
        "Sigma",
        "factor_exposure",
        "w_prev",
        "decision_date",
        "future_dates",
        "future_valid_length",
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
    context["rebalance_frequency"] = np.asarray(
        config.problem.rebalance_frequency, dtype=np.int64
    )
    context["factor_names"] = samples[0]["factor_names"]
    for key in (
        "factor_lower",
        "factor_upper",
        "industry_exposure",
        "industry_names",
        "industry_lower",
        "industry_upper",
    ):
        if key in samples[0]:
            context[key] = samples[0][key]
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
    rebalance_frequencies: Optional[Dict[str, int]] = None,
    execution_dates_by_split: Optional[
        Dict[str, Sequence[Union[str, pd.Timestamp]]]
    ] = None,
    allow_incomplete_future_splits: Optional[Sequence[str]] = None,
    protocol: str = "sit",
) -> Dict[str, Path]:
    """Build train/val/test archives from a Date-indexed price CSV.

    ``rebalance_frequencies`` optionally overrides the frequency per split.
    This is needed when the model is trained on daily decision samples but
    evaluated on a lower-frequency rebalancing schedule.
    """

    frame = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    protocol = str(protocol).lower()
    if protocol not in {"sit", "native"}:
        raise ValueError("protocol must be one of sit or native")
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

    if rebalance_frequencies is None:
        rebalance_frequencies = {
            split: config.problem.rebalance_frequency for split in split_ranges
        }
    unknown_splits = set(rebalance_frequencies).difference(split_ranges)
    if unknown_splits:
        raise ValueError(
            "rebalance_frequencies contains unknown splits: "
            f"{sorted(unknown_splits)}"
        )
    for split, frequency in rebalance_frequencies.items():
        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError(
                f"rebalance frequency for {split!r} must be a positive integer"
            )

    output_dir = Path(output_dir)
    output_paths: Dict[str, Path] = {}
    for split, (start_date, end_date) in split_ranges.items():
        split_frame = frame
        split_execution_dates = (execution_dates_by_split or {}).get(split)
        split_allow_incomplete = split in set(allow_incomplete_future_splits or ())
        if protocol == "sit":
            # Match data_factory.py: train/val use only their own rows, while
            # test receives W+1 rows of context before the test start.
            if split in {"train", "val"}:
                split_frame = frame.loc[start_date:end_date]
            elif split == "test":
                test_start_index = int(
                    frame.index.searchsorted(pd.Timestamp(start_date), side="left")
                )
                context_start_index = test_start_index - (
                    config.problem.lookback_window + 1
                )
                if context_start_index < 0:
                    raise ValueError("not enough pre-test history for SIT warm-up")
                split_frame = frame.iloc[context_start_index:].loc[:end_date]
                split_allow_incomplete = True

        split_config = replace(
            config,
            problem=replace(
                config.problem,
                rebalance_frequency=(
                    1 if protocol == "sit" else rebalance_frequencies[split]
                ),
            ),
        )
        context = build_contexts(
            split_frame,
            split_config,
            start_date if protocol != "sit" or split == "test" else None,
            end_date if protocol != "sit" or split == "test" else None,
            execution_dates=split_execution_dates,
            allow_incomplete_future=split_allow_incomplete,
        )
        path = output_dir / f"{split}.npz"
        save_context_cache(context, path)
        output_paths[split] = path
    return output_paths
