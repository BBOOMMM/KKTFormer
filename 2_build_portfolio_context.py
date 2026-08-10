"""Precompute point-in-time KKTFormer portfolio contexts.

Example:

    python 2_build_portfolio_context.py \
        --data_path ./asset_data/full_dataset.csv \
        --output_dir ./portfolio_context_cache/pool_30 \
        --data_pool 30 \
        --lookback_window 60 \
        --horizon 20 \
        --rebalance_frequency 20
"""

import argparse

from portfolio.context_builder import (
    PortfolioContextConfig,
    build_context_cache_from_csv,
    load_industry_exposure,
)
from portfolio.problem import MinimalPortfolioProblem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build point-in-time KKTFormer portfolio context caches"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./asset_data/full_dataset.csv",
        help="CSV with a Date column and price columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./portfolio_context_cache/pool_30",
        help="directory receiving train.npz, val.npz and test.npz",
    )
    parser.add_argument("--data_pool", type=int, default=30)
    parser.add_argument("--lookback_window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--rebalance_frequency", type=int, default=20)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--upper_bound", type=float, default=1.0)
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument("--budget_target", type=float, default=1.0)
    parser.add_argument("--covariance_epsilon", type=float, default=1e-6)
    parser.add_argument("--transaction_cost_bps", type=float, default=0.0)
    parser.add_argument("--factor_lower", type=str, default="")
    parser.add_argument("--factor_upper", type=str, default="")
    parser.add_argument("--industry_exposure_path", type=str, default="")
    parser.add_argument("--industry_lower", type=str, default="")
    parser.add_argument("--industry_upper", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    def parse_bound(value):
        if not value:
            return None
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if len(pieces) == 1:
            return float(pieces[0])
        return tuple(float(piece) for piece in pieces)

    problem = MinimalPortfolioProblem(
        num_assets=args.data_pool,
        lookback_window=args.lookback_window,
        horizon=args.horizon,
        rebalance_frequency=args.rebalance_frequency,
        eta=args.eta,
        upper_bound=args.upper_bound,
        lower_bound=args.lower_bound,
        budget_target=args.budget_target,
    )
    industry_exposure = None
    industry_names = ()
    if args.industry_exposure_path:
        import pandas as pd

        frame = pd.read_csv(args.data_path, parse_dates=["Date"])
        industry_exposure, industry_names = load_industry_exposure(
            args.industry_exposure_path,
            frame.columns.drop("Date"),
        )
    config = PortfolioContextConfig(
        problem=problem,
        covariance_epsilon=args.covariance_epsilon,
        transaction_cost_bps=args.transaction_cost_bps,
        factor_lower=parse_bound(args.factor_lower),
        factor_upper=parse_bound(args.factor_upper),
        industry_exposure=industry_exposure,
        industry_names=industry_names,
        industry_lower=parse_bound(args.industry_lower),
        industry_upper=parse_bound(args.industry_upper),
    )
    output_paths = build_context_cache_from_csv(
        csv_path=args.data_path,
        output_dir=args.output_dir,
        config=config,
        data_pool=args.data_pool,
    )
    for split, path in output_paths.items():
        print(f"[{split}] saved context cache: {path}")


if __name__ == "__main__":
    main()
