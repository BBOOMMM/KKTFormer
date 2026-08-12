"""Shared experiment protocol constants copied from the released SIT code.

This module is intentionally protocol-only.  KKTFormer may use its own
features, objective and optimizer, but its data ranges and test execution
calendar can be kept identical to SIT without importing the SIT experiment
implementation.
"""

from typing import Tuple


SIT_TRAIN_RANGE: Tuple[str, str] = ("2000-01-01", "2016-12-31")
SIT_VAL_RANGE: Tuple[str, str] = ("2017-01-01", "2019-12-31")
SIT_TEST_RANGE: Tuple[str, str] = ("2020-01-01", "2024-12-31")

SIT_SPLIT_RANGES = {
    "train": SIT_TRAIN_RANGE,
    "val": SIT_VAL_RANGE,
    "test": SIT_TEST_RANGE,
}

# Exact copy of REBAL_DATES_STR in exp/exp_main.py.  These are execution dates,
# not KKT decision dates: a KKT context for date R observes data through R-1.
SIT_REBALANCE_DATES = (
    "2020-01-03", "2020-01-31", "2020-03-02", "2020-03-30", "2020-04-28",
    "2020-05-27", "2020-06-24", "2020-07-23", "2020-08-20", "2020-09-18",
    "2020-10-16", "2020-11-13", "2020-12-14", "2021-01-13", "2021-02-11",
    "2021-03-12", "2021-04-12", "2021-05-10", "2021-06-08", "2021-07-07",
    "2021-08-04", "2021-09-01", "2021-09-30", "2021-10-28", "2021-11-26",
    "2021-12-27", "2022-01-25", "2022-02-23", "2022-03-23", "2022-04-21",
    "2022-05-19", "2022-06-17", "2022-07-19", "2022-08-16", "2022-09-14",
    "2022-10-12", "2022-11-09", "2022-12-08", "2023-01-09", "2023-02-07",
    "2023-03-08", "2023-04-05", "2023-05-04", "2023-06-02", "2023-07-03",
    "2023-08-01", "2023-08-29", "2023-09-27", "2023-10-25", "2023-11-22",
    "2023-12-21", "2024-01-23", "2024-02-21", "2024-03-20", "2024-04-18",
    "2024-05-16", "2024-06-14", "2024-07-16", "2024-08-13", "2024-09-11",
    "2024-10-09", "2024-11-06", "2024-12-05",
)

SIT_REBALANCE_DATE_SET = frozenset(SIT_REBALANCE_DATES)

