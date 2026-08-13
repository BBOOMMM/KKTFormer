#!/bin/bash

set -euo pipefail

# Reuse all hyperparameters from kkt_test_30.sh; only select the DOW10 data.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KKT_DATA_PATH="DOW30_ret.csv" \
KKT_CONTEXT_ROOT="./portfolio_context_cache_dow30" \
KKT_INPUT_KIND="returns" \
KKT_DATA_POOL=10 \
KKT_PROBE_UPPER_BOUND=0.15 \
KKT_MODEL_ID="kkt_dow10" \
bash "$script_dir/kkt_test_30.sh"
