"""Data provider for KKTFormer-v0 portfolio contexts."""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from data_provider.data_loader_kkt import Dataset_PortfolioContext
from portfolio.context_builder import PortfolioContextConfig, load_industry_exposure
from portfolio.problem import MinimalPortfolioProblem


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
        return _build_loader(dataset, flag, args)

    csv_path = os.path.join(args.root_path, args.data_path)
    prices = (
        pd.read_csv(csv_path, parse_dates=["Date"])
        .set_index("Date")
        .iloc[:, : args.data_pool]
    )
    problem = MinimalPortfolioProblem(
        num_assets=args.data_pool,
        lookback_window=args.window_size,
        horizon=args.horizon,
        rebalance_frequency=args.rebalance_frequency,
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
            industry_path, prices.columns
        )
    context_config = PortfolioContextConfig(
        problem=problem,
        covariance_epsilon=args.covariance_epsilon,
        transaction_cost_bps=args.transaction_cost_bps,
        factor_lower=_parse_bound_text(getattr(args, "factor_lower", "")),
        factor_upper=_parse_bound_text(getattr(args, "factor_upper", "")),
        industry_exposure=industry_exposure,
        industry_names=industry_names,
        industry_lower=_parse_bound_text(getattr(args, "industry_lower", "")),
        industry_upper=_parse_bound_text(getattr(args, "industry_upper", "")),
    )
    split_ranges = {
        "train": ("2000-01-01", "2016-12-31"),
        "val": ("2017-01-01", "2019-12-31"),
        "test": ("2020-01-01", "2024-12-31"),
    }
    pred_start, pred_end = split_ranges[flag]
    dataset = Dataset_PortfolioContext(
        prices=prices,
        config=context_config,
        pred_start=pred_start,
        pred_end=pred_end,
    )
    return _build_loader(dataset, flag, args)
