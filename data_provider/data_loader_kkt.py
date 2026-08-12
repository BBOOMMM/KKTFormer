"""Dataset for the stage-2 KKTFormer portfolio context."""

from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from portfolio.context_builder import (
    PortfolioContextConfig,
    build_contexts,
    load_context_cache,
)


class Dataset_PortfolioContext(Dataset):
    """Return market data and point-in-time optimization context.

    The dataset can either build contexts from a full price DataFrame or load
    a previously generated ``.npz`` split.  When date filtering is used, the
    full DataFrame is still passed to ``build_contexts`` so validation and test
    samples retain their historical lookback without using future observations.
    """

    def __init__(
        self,
        prices: Optional[pd.DataFrame] = None,
        config: Optional[PortfolioContextConfig] = None,
        pred_start: Optional[Union[str, pd.Timestamp]] = None,
        pred_end: Optional[Union[str, pd.Timestamp]] = None,
        execution_dates=None,
        allow_incomplete_future: bool = False,
        cache_path: Optional[Union[str, bytes]] = None,
    ) -> None:
        super().__init__()
        if (prices is None) == (cache_path is None):
            raise ValueError("provide exactly one of prices or cache_path")
        if prices is not None and config is None:
            raise ValueError("config is required when building contexts from prices")

        if cache_path is not None:
            self.context = load_context_cache(cache_path)
        else:
            self.context = build_contexts(
                prices,
                config,
                start_date=pred_start,
                end_date=pred_end,
                execution_dates=execution_dates,
                allow_incomplete_future=allow_incomplete_future,
            )

        self._validate_context()
        self.num_assets = int(self.context["log_return_path"].shape[-2])
        self.lookback_window = int(self.context["log_return_path"].shape[-1])
        self.horizon = int(self.context["future_returns"].shape[-2])

    def _validate_context(self) -> None:
        required = {
            "log_return_path",
            "date_feats",
            "future_returns",
            "Sigma",
            "factor_exposure",
            "w_prev",
            "decision_date",
            "future_dates",
            "lower_bounds",
            "upper_bounds",
            "budget_target",
            "transaction_cost_rate",
        }
        missing = required.difference(self.context)
        if missing:
            raise ValueError(f"context cache is missing fields: {sorted(missing)}")

        paths = self.context["log_return_path"]
        if paths.ndim != 4:
            raise ValueError("log_return_path must have shape (samples, H, N, W)")
        if self.context["date_feats"].shape[1:] != (paths.shape[1], 3):
            raise ValueError("date_feats must have shape (samples, H, 3)")
        n_samples, horizon, num_assets, _ = paths.shape
        if self.context["future_returns"].shape != (n_samples, horizon, num_assets):
            raise ValueError("future_returns must have shape (samples, H, N)")
        if self.context["Sigma"].shape != (
            n_samples,
            horizon,
            num_assets,
            num_assets,
        ):
            raise ValueError(
                "Sigma must have shape (samples, H, N, N); rebuild legacy "
                "context caches that contain one covariance per sample"
            )
        factor_exposure = self.context["factor_exposure"]
        if (
            factor_exposure.ndim != 4
            or factor_exposure.shape[:3] != (n_samples, horizon, num_assets)
        ):
            raise ValueError(
                "factor_exposure must have shape (samples, H, N, K); rebuild "
                "legacy context caches that contain one exposure per sample"
            )
        if self.context["w_prev"].shape != (n_samples, num_assets):
            raise ValueError("w_prev must have shape (samples, N)")
        for key in ("date_feats", "future_returns", "Sigma", "factor_exposure", "w_prev", "decision_date", "future_dates"):
            if self.context[key].shape[0] != n_samples:
                raise ValueError(f"context field {key} has inconsistent sample count")
        if n_samples == 0:
            raise ValueError("context must contain at least one sample")

    def __len__(self) -> int:
        return int(self.context["log_return_path"].shape[0])

    def __getitem__(self, index: int):
        context = self.context
        decision_date = pd.Timestamp(context["decision_date"][index]).strftime("%Y-%m-%d")
        future_dates = [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in context["future_dates"][index]
        ]
        item = {
            "log_return_path": torch.from_numpy(
                np.asarray(context["log_return_path"][index], dtype=np.float32)
            ),
            "date_feats": torch.from_numpy(
                np.asarray(context["date_feats"][index], dtype=np.float32)
            ),
            "future_returns": torch.from_numpy(
                np.asarray(context["future_returns"][index], dtype=np.float32)
            ),
            "Sigma": torch.from_numpy(
                np.asarray(context["Sigma"][index], dtype=np.float32)
            ),
            "factor_exposure": torch.from_numpy(
                np.asarray(context["factor_exposure"][index], dtype=np.float32)
            ),
            "w_prev": torch.from_numpy(
                np.asarray(context["w_prev"][index], dtype=np.float32)
            ),
            "lower_bounds": torch.from_numpy(
                np.asarray(context["lower_bounds"], dtype=np.float32)
            ),
            "upper_bounds": torch.from_numpy(
                np.asarray(context["upper_bounds"], dtype=np.float32)
            ),
            "budget_target": torch.as_tensor(
                context["budget_target"], dtype=torch.float32
            ),
            "transaction_cost_rate": torch.as_tensor(
                context["transaction_cost_rate"], dtype=torch.float32
            ),
            "decision_date": decision_date,
            "future_dates": future_dates,
        }
        if "future_valid_length" in context:
            item["future_valid_length"] = torch.as_tensor(
                context["future_valid_length"][index], dtype=torch.int64
            )
        optional_tensor_fields = (
            "factor_lower",
            "factor_upper",
            "industry_exposure",
            "industry_lower",
            "industry_upper",
        )
        for field in optional_tensor_fields:
            if field in context:
                item[field] = torch.from_numpy(
                    np.asarray(context[field], dtype=np.float32)
                )
        if "industry_names" in context:
            item["industry_names"] = [str(name) for name in context["industry_names"]]
        return item
