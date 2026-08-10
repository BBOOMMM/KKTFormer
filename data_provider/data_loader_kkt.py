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
            )

        self._validate_context()
        self.num_assets = int(self.context["market_window"].shape[-2])
        self.lookback_window = int(self.context["market_window"].shape[-3])
        self.horizon = int(self.context["future_returns"].shape[-2])

    def _validate_context(self) -> None:
        required = {
            "market_window",
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

        n_samples = self.context["market_window"].shape[0]
        for key in ("future_returns", "Sigma", "factor_exposure", "w_prev", "decision_date", "future_dates"):
            if self.context[key].shape[0] != n_samples:
                raise ValueError(f"context field {key} has inconsistent sample count")
        if n_samples == 0:
            raise ValueError("context must contain at least one sample")

    def __len__(self) -> int:
        return int(self.context["market_window"].shape[0])

    def __getitem__(self, index: int):
        context = self.context
        decision_date = pd.Timestamp(context["decision_date"][index]).strftime("%Y-%m-%d")
        future_dates = [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in context["future_dates"][index]
        ]
        return {
            "market_window": torch.from_numpy(
                np.asarray(context["market_window"][index], dtype=np.float32)
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
