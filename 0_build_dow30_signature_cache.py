#!/usr/bin/env python3
"""Build SIT-compatible signature caches from a wide daily-return CSV."""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from data_provider.data_loader import Dataset_Sig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("asset_data/DOW30_ret.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("signature_cache_dow30_6020"))
    parser.add_argument("--pools", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    return parser.parse_args()


def fit_sig_scaler(dataset):
    rows = []
    for index in trange(len(dataset), desc="fit scaler"):
        item = dataset[index]
        rows.append(np.concatenate((item["x_sigs"].ravel(), item["cross_sigs"].ravel())))
    return StandardScaler().fit(np.vstack(rows))


def save_split(dataset, split, output_dir):
    n_samples, horizon, n_assets = len(dataset), dataset.H, dataset.D
    output_dir.mkdir(parents=True, exist_ok=True)

    def memmap(name, shape):
        return np.memmap(output_dir / name, dtype="float32", mode="w+", shape=shape)

    x = memmap(f"{split}_x.npy", (n_samples, horizon, n_assets, 2))
    cross = memmap(f"{split}_cross.npy", (n_samples, horizon, n_assets, n_assets, 1))
    returns = memmap(f"{split}_ret.npy", (n_samples, horizon, n_assets))
    dates = []
    for index in trange(n_samples, desc=f"save {split}"):
        item = dataset[index]
        x[index] = item["x_sigs"]
        cross[index] = item["cross_sigs"]
        returns[index] = item["future_return_unscaled"]
        dates.append(item["dates_horizon"])
    x.flush()
    cross.flush()
    returns.flush()
    np.save(output_dir / f"{split}_dates.npy", np.asarray(dates, dtype=object))
    with (output_dir / f"{split}_meta.json").open("w") as handle:
        json.dump({"N": n_samples, "H": horizon, "D": n_assets}, handle)
    print(f"[{split.upper()}] N={n_samples}, H={horizon}, D={n_assets} -> {output_dir}")


def main():
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["Date"]).set_index("Date").sort_index()
    max_pool = max(args.pools)
    if max_pool > frame.shape[1]:
        raise ValueError(f"requested pool {max_pool}, but CSV has only {frame.shape[1]} value columns")
    returns_all = frame.iloc[:, :max_pool].astype(np.float64)
    if not np.isfinite(returns_all.to_numpy()).all():
        raise ValueError("selected return columns contain NaN or infinite values")
    if (returns_all.to_numpy() <= -1.0).any():
        raise ValueError("simple returns must be greater than -1")

    # A unit-valued synthetic starting portfolio is sufficient: compounding
    # exactly converts each later price ratio back to the supplied return.
    wealth_all = (1.0 + returns_all).cumprod()
    context_pad = args.window_size + 1
    test_context_start = (
        pd.Timestamp(args.test_start) - BDay(context_pad)
    ).strftime("%Y-%m-%d")

    for pool in args.pools:
        asset_names = returns_all.columns[:pool].tolist()
        returns = returns_all.iloc[:, :pool]
        wealth = wealth_all.iloc[:, :pool]
        split_frames = {
            "train": (wealth.loc["2000-01-01":"2016-12-31"], returns.loc["2000-01-01":"2016-12-31"]),
            "val": (wealth.loc["2017-01-01":"2019-12-31"], returns.loc["2017-01-01":"2019-12-31"]),
            "test": (wealth.loc[test_context_start:args.test_end], returns.loc[test_context_start:args.test_end]),
        }
        datasets = {}
        for split, (wealth_split, return_split) in split_frames.items():
            prediction_bounds = (
                {"pred_start": args.test_start, "pred_end": args.test_end}
                if split == "test" else {}
            )
            datasets[split] = Dataset_Sig(
                wealth_split,
                return_split,
                args.window_size,
                args.horizon,
                target_kind="returns",
                **prediction_bounds,
            )

        scaler = fit_sig_scaler(datasets["train"])
        for dataset in datasets.values():
            dataset.set_sig_scaler(scaler)
        output_dir = args.output_root / f"pool_{pool}"
        for split, dataset in datasets.items():
            save_split(dataset, split, output_dir)
        joblib.dump(scaler, output_dir / "signature_scaler.pkl")
        with (output_dir / "cache_info.json").open("w") as handle:
            json.dump(
                {
                    "source": str(args.input),
                    "input_kind": "simple_returns",
                    "target_convention": "date_t_to_t_plus_1_from_return_row_t_plus_1",
                    "signature_path": "compounded_wealth",
                    "assets": asset_names,
                    "date_start": str(returns.index.min().date()),
                    "date_end": str(returns.index.max().date()),
                    "window_size": args.window_size,
                    "horizon": args.horizon,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
