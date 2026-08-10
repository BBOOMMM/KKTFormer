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
    parser.add_argument("--covariance_epsilon", type=float, default=1e-6)
    parser.add_argument("--transaction_cost_bps", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    problem = MinimalPortfolioProblem(
        num_assets=args.data_pool,
        lookback_window=args.lookback_window,
        horizon=args.horizon,
        rebalance_frequency=args.rebalance_frequency,
        eta=args.eta,
        upper_bound=args.upper_bound,
    )
    config = PortfolioContextConfig(
        problem=problem,
        covariance_epsilon=args.covariance_epsilon,
        transaction_cost_bps=args.transaction_cost_bps,
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
