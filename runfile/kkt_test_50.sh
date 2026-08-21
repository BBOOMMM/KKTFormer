#!/bin/bash

set -euo pipefail

# Run this script from the KKTFormer repository root:
#   bash runfile/kkt_test_50.sh

export CUDA_VISIBLE_DEVICES="${KKT_CUDA_VISIBLE_DEVICES:-0}"

# -----------------------------------------------------------------------------
# Experiment paths and protocol
# -----------------------------------------------------------------------------
root_path="./asset_data/"
data_path="full_dataset.csv"
context_root="./portfolio_context_cache_robust60"
checkpoints="./checkpoints_kkt/"
results_path="./results_kkt_final/"
log_dir="./logs_kkt_final/"

protocol="sit"                    # sit | native
data_pool=50
window_size=60                   # fair SIT-aligned lookback window
horizon=20
rebalance_frequency=1             # ignored by the SIT protocol
train_rebalance_frequency=""      # empty: use rebalance_frequency
val_rebalance_frequency=""        # empty: use rebalance_frequency
test_rebalance_frequency=""       # empty: use rebalance_frequency
evaluation_end_date="2024-12-31" # used by the native protocol

# -----------------------------------------------------------------------------
# Portfolio problem, signal scaling, costs, and constraints
# -----------------------------------------------------------------------------
upper_bound=1.0
lower_bound=0.0
probe_upper_bound=0.0201          # nearly active KKT box geometry for N=50
probe_lower_bound=0.0
budget_target=1.0
eta=1e-3
covariance_epsilon=1e-6
covariance_robustness=0.35        # diagonal shrinkage after robustification
covariance_decay=0.98             # causal recency weighting
covariance_winsor_quantile=0.05   # causal tail clipping per asset

signal_normalization="risk"       # risk | none
signal_scale=0.12
signal_normalization_epsilon=1e-6

trade_cost_bps=0.0
transaction_cost_bps=""           # empty: use trade_cost_bps
transaction_cost_smoothing=1e-4
turnover_penalty=0.02             # smooth path-level turnover objective
mean_return_weight="${KKT_MEAN_RETURN_WEIGHT:-0.0}"
turnover_smoothing=1.0            # preserve the first causal risk-budget decision

# Entropy regularization in the optimizer and KKT state:
#   tau * sum_i [(w_i + epsilon) log(w_i + epsilon) - epsilon log(epsilon)]
# tau=0 disables it; positive tau encourages a more diversified portfolio.
# Supported by every feedback mode.
entropy_regularization="${KKT_ENTROPY_REGULARIZATION:-0.001}"
entropy_epsilon=1e-4              # numerical smoothing near w_i=0

max_turnover=""                    # risk-budget allocator uses causal policy proposals
gross_exposure_limit=""           # e.g. 1.0; empty disables it
factor_lower=""
factor_upper=""
industry_exposure_path=""         # empty disables industry constraints
industry_lower=""                 # scalar or comma-separated bounds
industry_upper=""                 # scalar or comma-separated bounds
sequential_state=0                # 1 adds --sequential_state

# -----------------------------------------------------------------------------
# KKTFormer input projections and Transformer
# -----------------------------------------------------------------------------
input_dim=1
factor_dim=3
feedback_mode="dual"             # KKT context and low-rank bias paths together
decision_layer="${KKT_DECISION_LAYER:-risk_budget}"
active_tolerance=1e-5

log_return_embed_dim=32
date_embed_dim=8
asset_embed_dim=8
d_model=32                        # fusion output and Transformer hidden width
n_heads=4
num_layers=1
ff_dim=64
dropout=0.0

# -----------------------------------------------------------------------------
# Differentiable optimizer and objective
# -----------------------------------------------------------------------------
optimizer_iterations=100
probe_optimizer_iterations=30
projection_iterations=64
constraint_projection_iterations=20

loss_mode="${KKT_LOSS_MODE:-ktr}" # CVaR + detached decision-regret + KTR
regret_weight="${KKT_REGRET_WEIGHT:-0.01}" # lambda_regret; oracle is not a prediction loss
prediction_loss="NONE"             # kept only for CLI compatibility; never used
prediction_weight=0.0
cvar_alpha=0.95
cvar_variant="sit"                # sit | smooth
cvar_temperature=1e-3
ktr_weight="${KKT_KTR_WEIGHT:-0.0001}"
ktr_tail_alpha="${KKT_KTR_TAIL_ALPHA:-0.80}"
ktr_pressure_scale="${KKT_KTR_PRESSURE_SCALE:-1.0}"
ktr_ranking_temperature="${KKT_KTR_RANKING_TEMPERATURE:-1.0}"
ktr_pressure_clip="${KKT_KTR_PRESSURE_CLIP:-5.0}"
kkt_bias_rank=3
temperature="${KKT_TEMPERATURE:-1.0}" # avoid GPU-sensitive near-corner allocation jumps
simplex_anchor_weight="${KKT_SIMPLEX_ANCHOR_WEIGHT:-0.0}"
momentum_anchor_weight="${KKT_MOMENTUM_ANCHOR_WEIGHT:-0.0}"
momentum_anchor_lookback="${KKT_MOMENTUM_ANCHOR_LOOKBACK:-20}"
momentum_anchor_temperature="${KKT_MOMENTUM_ANCHOR_TEMPERATURE:-1.0}"
risk_momentum_lookback=60         # do not introduce a hidden >60-day lookback
risk_scale_windows="20,40,60"        # trainable multi-scale gate within W=60
risk_score_normalization="raw"   # preserve return/volatility scale for temperature-aware allocation
risk_score_epsilon=1e-4           # stabilizes the raw risk score without flattening its scale
risk_multiscale_residual_weight=0.001 # bounded correction around the 20d base scale
risk_defensive_gate_floor=0.95      # small but genuine learned downside modulation
risk_momentum_short_weight="${KKT_RISK_SHORT_WEIGHT:-0.15}" # short-horizon trend correction
risk_momentum_residual_weight="${KKT_RISK_RESIDUAL_WEIGHT:-0.05}" # bounded learned Transformer allocation residual
risk_gate_logit_scale="${KKT_GATE_LOGIT_SCALE:-0.1}" # keep route gates near the causal prior
risk_forecast_weight="${KKT_FORECAST_WEIGHT:-0.0}" # bounded learned return route
risk_prior_bias=-0.8                # former successful baseline
risk_turnover_aversion=0.0
risk_downside_weight=0.25
risk_drawdown_weight=0.10
risk_smoothing_temperature=0.01
kkt_risk_scale=0.001               # isolated nonzero direct KKT-conditioned feedback
forecast_weight=0.0                # prediction loss is explicitly disabled
risk_contrarian_weight="${KKT_RISK_CONTRARIAN_WEIGHT:-0.0}" # isolate the learned trend/KKT interaction
risk_defensive_weight="${KKT_RISK_DEFENSIVE_WEIGHT:-1.0}" # learned downside-risk budget correction

# -----------------------------------------------------------------------------
# Training and hardware
# -----------------------------------------------------------------------------
learning_rate="${KKT_LEARNING_RATE:-1e-3}"
lradj="type1"
train_epochs="${KKT_TRAIN_EPOCHS:-2}" # decision-policy training, no prediction pretext task
ema_decay="${KKT_EMA_DECAY:-0.0}"
batch_size="${KKT_BATCH_SIZE:-128}"
patience="${KKT_PATIENCE:-4}"
num_workers=0
seed="${KKT_SEED:-2023,2024,2025}"

use_gpu=1
gpu="${KKT_GPU:-0}"
use_multi_gpu=0                   # 1 adds --use_multi_gpu
devices="0,1"

# Encode the main architecture choices in the experiment name. run_kkt.py also
# appends the remaining protocol/optimizer/loss settings to its checkpoint key.
model_id="${KKT_MODEL_ID:-kkt_decision_regret_factor_turnover_dp${data_pool}_w${window_size}}"

cmd=(
  conda run --no-capture-output -n alden python -u run_kkt.py
  --model_id "$model_id"
  --root_path "$root_path"
  --data_path "$data_path"
  --context_root "$context_root"
  --checkpoints "$checkpoints"
  --results_path "$results_path"
  --log_dir "$log_dir"
  --protocol "$protocol"
  --data_pool "$data_pool"
  --window_size "$window_size"
  --horizon "$horizon"
  --rebalance_frequency "$rebalance_frequency"
  --evaluation_end_date "$evaluation_end_date"
  --upper_bound "$upper_bound"
  --lower_bound "$lower_bound"
  --probe_upper_bound "$probe_upper_bound"
  --probe_lower_bound "$probe_lower_bound"
  --budget_target "$budget_target"
  --eta "$eta"
  --covariance_epsilon "$covariance_epsilon"
  --covariance_robustness "$covariance_robustness"
  --covariance_decay "$covariance_decay"
  --covariance_winsor_quantile "$covariance_winsor_quantile"
  --signal_normalization "$signal_normalization"
  --signal_scale "$signal_scale"
  --signal_normalization_epsilon "$signal_normalization_epsilon"
  --trade_cost_bps "$trade_cost_bps"
  --transaction_cost_smoothing "$transaction_cost_smoothing"
  --turnover_penalty "$turnover_penalty"
  --mean_return_weight "$mean_return_weight"
  --turnover_smoothing "$turnover_smoothing"
  --risk_turnover_aversion "$risk_turnover_aversion"
  --entropy_regularization "$entropy_regularization"
  --entropy_epsilon "$entropy_epsilon"
  "--factor_lower=$factor_lower"
  "--factor_upper=$factor_upper"
  --industry_exposure_path "$industry_exposure_path"
  "--industry_lower=$industry_lower"
  "--industry_upper=$industry_upper"
  --input_dim "$input_dim"
  --factor_dim "$factor_dim"
  --feedback_mode "$feedback_mode"
  --decision_layer "$decision_layer"
  --active_tolerance "$active_tolerance"
  --log_return_embed_dim "$log_return_embed_dim"
  --date_embed_dim "$date_embed_dim"
  --asset_embed_dim "$asset_embed_dim"
  --d_model "$d_model"
  --n_heads "$n_heads"
  --num_layers "$num_layers"
  --ff_dim "$ff_dim"
  --dropout "$dropout"
  --optimizer_iterations "$optimizer_iterations"
  --probe_optimizer_iterations "$probe_optimizer_iterations"
  --projection_iterations "$projection_iterations"
  --constraint_projection_iterations "$constraint_projection_iterations"
  --loss_mode "$loss_mode"
  --regret_weight "$regret_weight"
  --prediction_loss "$prediction_loss"
  --forecast_weight "$forecast_weight"
  --cvar_alpha "$cvar_alpha"
  --cvar_variant "$cvar_variant"
  --cvar_temperature "$cvar_temperature"
  --ktr_weight "$ktr_weight"
  --ktr_tail_alpha "$ktr_tail_alpha"
  --ktr_pressure_scale "$ktr_pressure_scale"
  --ktr_ranking_temperature "$ktr_ranking_temperature"
  --ktr_pressure_clip "$ktr_pressure_clip"
  --kkt_bias_rank "$kkt_bias_rank"
  --kkt_risk_scale "$kkt_risk_scale"
  --prediction_weight "$prediction_weight"
  --temperature "$temperature"
  --simplex_anchor_weight "$simplex_anchor_weight"
  --momentum_anchor_weight "$momentum_anchor_weight"
  --momentum_anchor_lookback "$momentum_anchor_lookback"
  --momentum_anchor_temperature "$momentum_anchor_temperature"
  --risk_momentum_lookback "$risk_momentum_lookback"
  --risk_scale_windows "$risk_scale_windows"
  --risk_score_normalization "$risk_score_normalization"
  --risk_score_epsilon "$risk_score_epsilon"
  --risk_multiscale_residual_weight "$risk_multiscale_residual_weight"
  --risk_defensive_gate_floor "$risk_defensive_gate_floor"
  --risk_momentum_short_weight "$risk_momentum_short_weight"
  --risk_momentum_residual_weight "$risk_momentum_residual_weight"
  --risk_gate_logit_scale "$risk_gate_logit_scale"
  --risk_forecast_weight "$risk_forecast_weight"
  --risk_contrarian_weight "$risk_contrarian_weight"
  --risk_defensive_weight "$risk_defensive_weight"
  --risk_prior_bias "$risk_prior_bias"
  --risk_downside_weight "$risk_downside_weight"
  --risk_drawdown_weight "$risk_drawdown_weight"
  --risk_smoothing_temperature "$risk_smoothing_temperature"
  --learning_rate "$learning_rate"
  --lradj "$lradj"
  --train_epochs "$train_epochs"
  --ema_decay "$ema_decay"
  --batch_size "$batch_size"
  --patience "$patience"
  --num_workers "$num_workers"
  --seed "$seed"
  --use_gpu "$use_gpu"
  --gpu "$gpu"
  --devices "$devices"
)

# argparse options whose disabled value is None are omitted when left empty.
[[ -n "$train_rebalance_frequency" ]] && cmd+=(--train_rebalance_frequency "$train_rebalance_frequency")
[[ -n "$val_rebalance_frequency" ]] && cmd+=(--val_rebalance_frequency "$val_rebalance_frequency")
[[ -n "$test_rebalance_frequency" ]] && cmd+=(--test_rebalance_frequency "$test_rebalance_frequency")
[[ -n "$transaction_cost_bps" ]] && cmd+=(--transaction_cost_bps "$transaction_cost_bps")
[[ -n "$max_turnover" ]] && cmd+=(--max_turnover "$max_turnover")
[[ -n "$gross_exposure_limit" ]] && cmd+=(--gross_exposure_limit "$gross_exposure_limit")

(( sequential_state == 1 )) && cmd+=(--sequential_state)
(( use_multi_gpu == 1 )) && cmd+=(--use_multi_gpu)

echo "Running with model_id: $model_id"
printf 'Command:'
printf ' %s' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
