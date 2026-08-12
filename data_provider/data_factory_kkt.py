"""Data provider for KKTFormer-v0 portfolio contexts."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data_provider.data_loader_kkt import Dataset_PortfolioContext
from portfolio.context_builder import PortfolioContextConfig, load_industry_exposure
from portfolio.problem import MinimalPortfolioProblem
from utils.sit_protocol import SIT_SPLIT_RANGES


def _context_cache_dir(args) -> Optional[Path]:
    root = getattr(args, "context_root", "")
    if not root:
        return None
    root = Path(root)
    pool_dir = root / f"pool_{args.data_pool}"
    if pool_dir.is_dir():
        return pool_dir
    return root


def _build_loader(dataset, flag, args):
    sequential = bool(getattr(args, "sequential_state", False))
    if sequential:
        # A stateful portfolio path must be consumed in chronological order.
        batch_size = 1
        shuffle = False
        drop_last = False
    elif flag == "train":
        batch_size = args.batch_size
        shuffle = True
        drop_last = False
    elif flag == "val":
        batch_size = args.batch_size
        shuffle = False
        drop_last = False
    elif flag == "test":
        # Keeping test batches at one makes date/position export deterministic.
        batch_size = 1
        shuffle = False
        drop_last = False
    else:
        raise ValueError(f"unknown data split: {flag}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=getattr(args, "use_gpu", False),
    )
    print(f"[{flag.upper()}] len={len(dataset)} | batch_size={batch_size}")
    return dataset, loader


def _split_rebalance_frequency(args, flag):
    """Return the configured decision frequency for one data split."""

    if str(getattr(args, "protocol", "sit")).lower() == "sit":
        # SIT constructs daily train/validation samples.  Test decisions are
        # selected from the fixed execution calendar below instead of by a
        # regular frequency.
        return 1

    override = getattr(args, f"{flag}_rebalance_frequency", None)
    return int(
        override
        if override is not None
        else getattr(args, "rebalance_frequency", 20)
    )


def _parse_bound_text(value):
    """Parse ``-0.1,0.1``/scalar CLI bounds while preserving ``None``."""

    if value is None or value == "":
        return None
    if isinstance(value, (float, int)):
        return float(value)
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if len(pieces) == 1:
        return float(pieces[0])
    return tuple(float(piece) for piece in pieces)


def data_provider_kkt(args, flag):
    """Return a KKTFormer context dataset and loader.

    When ``context_root`` points to an existing cache, the split is loaded
    directly.  Otherwise contexts are built from the full price DataFrame so
    validation and test samples retain their historical lookback.
    """

    cache_dir = _context_cache_dir(args)
    cache_path = cache_dir / f"{flag}.npz" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        dataset = Dataset_PortfolioContext(cache_path=cache_path)
        if str(getattr(args, "protocol", "sit")).lower() == "sit":
            frequency = np.asarray(dataset.context.get("rebalance_frequency", 1))
            decision_indices = np.asarray(
                dataset.context.get("decision_index", ()), dtype=np.int64
            )
            is_daily = (
                decision_indices.size > 0
                and (
                    decision_indices.size == 1
                    or np.all(np.diff(decision_indices) == 1)
                )
            )
            if (
                not np.all(frequency == 1)
                or "future_valid_length" not in dataset.context
                or not is_daily
            ):
                raise ValueError(
                    "the context cache is not SIT-compatible; rebuild it with "
                    "2_build_portfolio_context.py --protocol sit"
                )
        return _build_loader(dataset, flag, args)

    csv_path = os.path.join(args.root_path, args.data_path)
    combined_prices = (
        pd.read_csv(csv_path, parse_dates=["Date"])
        .set_index("Date")
        .iloc[:, : args.data_pool]
    )
    protocol = str(getattr(args, "protocol", "sit")).lower()

    if protocol == "sit":
        # Match SIT's split construction.  Train and validation do not borrow
        # historical rows from the previous split; test receives exactly the
        # W+1-row context warm-up used in data_factory.py.
        split_ranges = SIT_SPLIT_RANGES
        train_start, train_end = split_ranges["train"]
        val_start, val_end = split_ranges["val"]
        test_start, test_end = split_ranges["test"]
        test_start_index = int(
            combined_prices.index.searchsorted(pd.Timestamp(test_start), side="left")
        )
        context_start_index = test_start_index - (args.window_size + 1)
        if context_start_index < 0:
            raise ValueError("not enough pre-test history for the SIT context warm-up")
        split_prices = {
            "train": combined_prices.loc[train_start:train_end],
            "val": combined_prices.loc[val_start:val_end],
            "test": combined_prices.iloc[context_start_index:].loc[:test_end],
        }
        df_use = split_prices[flag]
    else:
        split_ranges = {
            "train": ("2000-01-01", "2016-12-31"),
            "val": ("2017-01-01", "2019-12-31"),
            "test": ("2020-01-01", "2024-12-31"),
        }
        df_use = combined_prices
    problem = MinimalPortfolioProblem(
        num_assets=args.data_pool,
        lookback_window=args.window_size,
        horizon=args.horizon,
        rebalance_frequency=_split_rebalance_frequency(args, flag),
        eta=args.eta,
        upper_bound=args.upper_bound,
        lower_bound=getattr(args, "lower_bound", 0.0),
        budget_target=getattr(args, "budget_target", 1.0),
    )
    industry_exposure = None
    industry_names = ()
    industry_path = getattr(args, "industry_exposure_path", "")
    if industry_path:
        industry_exposure, industry_names = load_industry_exposure(
            industry_path, combined_prices.columns
        )
    context_config = PortfolioContextConfig(
        problem=problem,
        covariance_epsilon=args.covariance_epsilon,
        transaction_cost_bps=float(
            getattr(args, "trade_cost_bps", 0.0)
            if getattr(args, "transaction_cost_bps", None) is None
            else args.transaction_cost_bps
        ),
        factor_lower=_parse_bound_text(getattr(args, "factor_lower", "")),
        factor_upper=_parse_bound_text(getattr(args, "factor_upper", "")),
        industry_exposure=industry_exposure,
        industry_names=industry_names,
        industry_lower=_parse_bound_text(getattr(args, "industry_lower", "")),
        industry_upper=_parse_bound_text(getattr(args, "industry_upper", "")),
    )
    pred_start, pred_end = split_ranges[flag]
    if protocol == "sit" and flag == "test":
        # SIT has 1237 test windows, each producing H daily predictions.  Their
        # union is a continuous daily prediction/return grid (1257 dates for
        # the released W=60, H=20 data).  KKTFormer emits one decision per
        # context, so construct that same effective daily grid directly.
        dataset = Dataset_PortfolioContext(
            prices=df_use,
            config=context_config,
            pred_start=pred_start,
            pred_end=pred_end,
            allow_incomplete_future=True,
        )
    elif protocol == "sit":
        dataset = Dataset_PortfolioContext(
            prices=df_use,
            config=context_config,
        )
    else:
        dataset = Dataset_PortfolioContext(
            prices=df_use,
            config=context_config,
            pred_start=pred_start,
            pred_end=pred_end,
        )
    return _build_loader(dataset, flag, args)
