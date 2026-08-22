"""Command-line entry point for KKTFormer-v0."""

import argparse
import hashlib
from pprint import pformat
import random

import numpy as np
import torch

from exp.exp_main_kkt import EXP_KKT, ensure_feasible_probe_upper_bound
from utils.logger import setup_logger


def parse_seed_list(value: str) -> list[int]:
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("--seed must contain at least one integer seed")
    try:
        return [int(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--seed expects an integer or comma-separated integers, e.g. 2023,2024,2025"
        ) from exc


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Keep CUDA kernels reproducible while comparing seeds.  We deliberately
    # avoid ``torch.use_deterministic_algorithms(True)`` here because some
    # optional constrained-optimizer kernels do not provide deterministic
    # implementations on every supported PyTorch/CUDA pair.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filesystem_safe_setting(setting: str, max_bytes: int = 240) -> str:
    """Compact an overlong experiment directory component deterministically."""

    encoded = setting.encode("utf-8")
    if len(encoded) <= max_bytes:
        return setting
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    marker = f"_h{digest}_"
    marker_bytes = len(marker.encode("utf-8"))
    available = max_bytes - marker_bytes
    prefix_budget = available // 2
    suffix_budget = available - prefix_budget
    prefix = encoded[:prefix_budget].decode("utf-8", errors="ignore")
    suffix = encoded[-suffix_budget:].decode("utf-8", errors="ignore")
    return f"{prefix}{marker}{suffix}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train KKTFormer-v0")
    parser.add_argument("--model_id", type=str, default="kkt_v0")
    parser.add_argument("--root_path", type=str, default="./asset_data/")
    parser.add_argument("--data_path", type=str, default="full_dataset.csv")
    parser.add_argument(
        "--input_kind",
        choices=["prices", "returns"],
        default="prices",
        help="interpret selected CSV value columns as prices or daily simple returns",
    )
    parser.add_argument("--context_root", type=str, default="")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints_kkt/")
    parser.add_argument("--results_path", type=str, default="./results_kkt/")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="directory for timestamped experiment log files",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["sit", "native"],
        default="sit",
        help="experiment protocol; sit matches the released SIT split/calendar",
    )
    parser.add_argument("--data_pool", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--rebalance_frequency",
        type=int,
        default=1,
        help="legacy regular context frequency; ignored by the SIT protocol",
    )
    parser.add_argument(
        "--train_rebalance_frequency",
        type=int,
        default=None,
        help="context frequency for train; falls back to rebalance_frequency",
    )
    parser.add_argument(
        "--val_rebalance_frequency",
        type=int,
        default=None,
        help="context frequency for validation; falls back to rebalance_frequency",
    )
    parser.add_argument(
        "--test_rebalance_frequency",
        type=int,
        default=None,
        help="context frequency for test; falls back to rebalance_frequency",
    )
    parser.add_argument(
        "--evaluation_end_date",
        type=str,
        default="2024-12-31",
        help="last date retained by the native test evaluator",
    )
    parser.add_argument("--upper_bound", type=float, default=1.0)
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument(
        "--probe_upper_bound",
        type=float,
        default=0.1,
        help="per-asset upper bound used only by the structural probe optimizer",
    )
    parser.add_argument(
        "--probe_lower_bound",
        type=float,
        default=0.0,
        help="per-asset lower bound used only by the structural probe optimizer",
    )
    parser.add_argument("--budget_target", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--covariance_epsilon", type=float, default=1e-6)
    parser.add_argument("--covariance_robustness", type=float, default=0.35)
    parser.add_argument("--covariance_decay", type=float, default=0.98)
    parser.add_argument("--covariance_winsor_quantile", type=float, default=0.05)
    parser.add_argument(
        "--signal_normalization",
        type=str,
        choices=["risk", "none"],
        default="risk",
        help="cross-sectionally normalize alpha and match it to quadratic-risk scale",
    )
    parser.add_argument(
        "--signal_scale",
        type=float,
        default=0.05,
        help="fixed c in tau_mu = c * mean(diag(Sigma + eta I))",
    )
    parser.add_argument(
        "--signal_normalization_epsilon", type=float, default=1e-6
    )
    parser.add_argument(
        "--trade_cost_bps",
        type=float,
        default=0.0,
        help="SIT-compatible transaction cost in basis points",
    )
    parser.add_argument(
        "--transaction_cost_bps",
        type=float,
        default=None,
        help="legacy alias; overrides --trade_cost_bps when supplied",
    )
    parser.add_argument("--transaction_cost_smoothing", type=float, default=1e-4)
    parser.add_argument("--turnover_penalty", type=float, default=0.0)
    parser.add_argument(
        "--mean_return_weight",
        type=float,
        default=0.0,
        help=(
            "portfolio-level mean-return utility weight in "
            "CVaR - weight * mean(sum_i w_i r_i); no asset prediction target"
        ),
    )
    parser.add_argument("--sortino_weight", type=float, default=0.0)
    parser.add_argument("--sharpe_weight", type=float, default=0.0)
    parser.add_argument(
        "--portfolio_statistic_scope",
        type=str,
        default="context",
        choices=["context", "batch"],
        help="estimate CVaR and return ratios per context or across a mini-batch",
    )
    parser.add_argument("--sortino_temperature", type=float, default=1e-2)
    parser.add_argument("--sortino_epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--turnover_smoothing",
        type=float,
        default=1.0,
        help="EMA step for sequential risk-budget allocations; lower means smoother turnover",
    )
    parser.add_argument(
        "--risk_turnover_aversion",
        type=float,
        default=0.0,
        help="proximal shrinkage strength of risk_budget allocations",
    )
    parser.add_argument(
        "--entropy_regularization",
        type=float,
        default=0.0,
        help="tau for tau * sum_i w_i log(w_i); 0 disables entropy regularization",
    )
    parser.add_argument(
        "--entropy_epsilon",
        type=float,
        default=1e-4,
        help="positive smoothing floor used when evaluating log(weights)",
    )
    parser.add_argument("--max_turnover", type=float, default=None)
    parser.add_argument("--gross_exposure_limit", type=float, default=None)
    parser.add_argument("--factor_lower", type=str, default="")
    parser.add_argument("--factor_upper", type=str, default="")
    parser.add_argument("--industry_exposure_path", type=str, default="")
    parser.add_argument("--industry_lower", type=str, default="")
    parser.add_argument("--industry_upper", type=str, default="")
    parser.add_argument(
        "--sequential_state",
        action="store_true",
        help="consume decisions chronologically and feed generated w_prev forward",
    )

    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--factor_dim", type=int, default=3)
    parser.add_argument(
        "--feedback_mode",
        type=str,
        choices=["none", "two_pass", "context", "bias", "dynamic", "dual", "jacobian"],
        default="dual",
        help=(
            "feedback ablation: none=one-pass, two_pass=matched refinement "
            "without KKT, context/bias=single KKT path, dynamic=hidden-conditioned "
            "KKT bias, dual=both KKT paths"
        ),
    )
    parser.add_argument("--active_tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--log_return_embed_dim",
        type=int,
        default=32,
        help="output width of the log-return path projection",
    )
    parser.add_argument(
        "--date_embed_dim",
        type=int,
        default=32,
        help="output width of the date-feature projection",
    )
    parser.add_argument(
        "--asset_embed_dim",
        type=int,
        default=32,
        help="width of the learned asset embedding",
    )
    parser.add_argument(
        "--asset_embedding_scale",
        type=float,
        default=1.0,
        help=(
            "scale of learned asset identity embeddings; 0 gives a "
            "permutation-equivariant ablation"
        ),
    )
    parser.add_argument(
        "--asset_embedding_init",
        type=str,
        choices=["random", "deterministic", "orthogonal"],
        default="random",
        help="trainable asset-code initialization; deterministic removes seed relabelling",
    )
    parser.add_argument(
        "--model_init_seed",
        type=int,
        default=-1,
        help=(
            "fixed seed used only for model parameter initialization; negative "
            "preserves seed-dependent initialization"
        ),
    )
    parser.add_argument(
        "--portfolio_heads",
        type=int,
        default=1,
        help="number of fully learned simplex policy experts",
    )
    parser.add_argument(
        "--policy_experts",
        type=int,
        default=1,
        help="number of complete decision-aware experts trained through one portfolio loss",
    )
    parser.add_argument(
        "--portfolio_aggregation",
        type=str,
        choices=["probability_mean", "logit_mean"],
        default="probability_mean",
        help="fuse learned policy experts before or after the simplex map",
    )
    parser.add_argument(
        "--policy_head_consistency_weight",
        type=float,
        default=0.0,
        help="label-free Jensen-Shannon consistency weight across policy experts",
    )
    parser.add_argument("--spectral_policy_filters", type=int, default=0)
    parser.add_argument("--spectral_policy_hidden", type=int, default=16)
    parser.add_argument("--spectral_policy_scale", type=float, default=1.0)
    parser.add_argument(
        "--tail_policy_filters",
        type=int,
        default=0,
        help="number of learnable distributional path queries; 0 disables the branch",
    )
    parser.add_argument("--tail_policy_hidden", type=int, default=16)
    parser.add_argument("--tail_policy_scale", type=float, default=1.0)
    parser.add_argument("--tail_policy_windows", type=str, default="5,10,20,60")
    parser.add_argument(
        "--ordered_policy_bins",
        type=int,
        default=8,
        help="number of learned empirical-quantile coordinates per lookback",
    )
    parser.add_argument(
        "--ordered_policy_scale",
        type=float,
        default=0.0,
        help="scale of the end-to-end learned ordered-distribution logits",
    )
    parser.add_argument(
        "--ordered_policy_windows", type=str, default="5,10,20,60"
    )
    parser.add_argument("--policy_refinement_steps", type=int, default=0)
    parser.add_argument("--policy_refinement_scale", type=float, default=0.0)
    parser.add_argument("--policy_refinement_window", type=int, default=60)
    parser.add_argument(
        "--policy_refinement_risk",
        choices=["variance", "cvar", "lpm"],
        default="variance",
    )
    parser.add_argument(
        "--policy_refinement_tail_temperature", type=float, default=0.01
    )
    parser.add_argument("--lpm_geometry_scale", type=float, default=0.0)
    parser.add_argument("--lpm_geometry_window", type=int, default=10)
    parser.add_argument("--lpm_geometry_init", type=float, default=2.5)
    parser.add_argument("--lpm_geometry_epsilon", type=float, default=1e-4)
    parser.add_argument("--relation_attention_scale", type=float, default=0.0)
    parser.add_argument("--relation_attention_hidden", type=int, default=16)
    parser.add_argument(
        "--use_asset_policy_bias",
        type=int,
        choices=[0, 1],
        default=0,
        help="zero-start asset memory learned only through the portfolio objective",
    )
    parser.add_argument("--asset_policy_bias_scale", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--optimizer_iterations",
        type=int,
        default=10,
        help="warm-started final optimizer steps; verify convergence for each asset pool",
    )
    parser.add_argument("--probe_optimizer_iterations", type=int, default=5)
    parser.add_argument(
        "--decision_layer",
        type=str,
        choices=["softmax", "optimizer", "risk_budget", "risk_optimizer"],
        default="softmax",
        help=(
            "final portfolio layer; risk_budget combines multi-scale "
            "risk-momentum logits with a turnover-aware allocator, while "
            "risk_optimizer feeds the same signal into the constrained QP"
        ),
    )
    parser.add_argument("--projection_iterations", type=int, default=64)
    parser.add_argument("--constraint_projection_iterations", type=int, default=20)
    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["cvar", "hybrid", "ktr", "risk_budget"],
        default="cvar",
        help=(
            "CVaR alone, CVaR plus decision regret, KKT tail ranking, or "
            "smooth downside/drawdown risk-budget loss"
        ),
    )
    parser.add_argument(
        "--regret_weight",
        type=float,
        default=0.1,
        help="lambda_regret in L = L_CVaR + lambda_regret * L_regret",
    )
    parser.add_argument("--prediction_loss", type=str, default="MSE")
    parser.add_argument("--cvar_alpha", type=float, default=0.95)
    parser.add_argument(
        "--cvar_variant",
        type=str,
        choices=["sit", "smooth"],
        default="sit",
        help="SIT-exact quantile/ReLU objective or detached-VaR smooth ablation",
    )
    parser.add_argument("--cvar_temperature", type=float, default=1e-3)
    parser.add_argument(
        "--ktr_weight",
        type=float,
        default=0.01,
        help="lambda_KTR in L = L_CVaR + lambda_KTR * L_KTR",
    )
    parser.add_argument(
        "--ktr_tail_alpha",
        type=float,
        default=0.95,
        help="tail quantile used to select KTR ranking scenarios",
    )
    parser.add_argument(
        "--ktr_pressure_scale",
        type=float,
        default=1.0,
        help="kappa controlling detached KKT pressure pair weights",
    )
    parser.add_argument("--ktr_ranking_temperature", type=float, default=1.0)
    parser.add_argument("--ktr_pressure_clip", type=float, default=5.0)
    parser.add_argument("--kkt_bias_rank", type=int, default=4)
    parser.add_argument(
        "--kkt_risk_scale",
        type=float,
        default=0.0,
        help="direct normalized KKT adjustment scale in risk_budget logits",
    )
    parser.add_argument(
        "--prediction_weight",
        type=float,
        default=0.0,
        help="deprecated compatibility option; prediction supervision is disabled",
    )
    parser.add_argument(
        "--forecast_weight",
        type=float,
        default=0.0,
        help="must be 0; cross-sectional return prediction loss is disabled",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="fixed temperature of the final softmax allocation head",
    )
    parser.add_argument(
        "--simplex_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "differentiable shrinkage of the final long-only policy toward "
            "the equal-weight simplex anchor"
        ),
    )
    parser.add_argument(
        "--momentum_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "differentiable residual-policy shrinkage toward a causal "
            "momentum simplex prior"
        ),
    )
    parser.add_argument("--momentum_anchor_lookback", type=int, default=20)
    parser.add_argument("--momentum_anchor_temperature", type=float, default=1.3)
    parser.add_argument(
        "--downside_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "differentiable shrinkage toward a causal inverse-downside-risk "
            "simplex prior"
        ),
    )
    parser.add_argument("--downside_anchor_lookback", type=int, default=10)
    parser.add_argument("--downside_anchor_power", type=float, default=2.0)
    parser.add_argument("--downside_anchor_epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--risk_momentum_lookback",
        type=int,
        default=60,
        help="number of genuine return observations used by the risk prior",
    )
    parser.add_argument(
        "--risk_scale_windows",
        type=str,
        default="20,40,60",
        help="comma-separated causal sub-windows for learned risk-scale fusion",
    )
    parser.add_argument(
        "--risk_score_normalization",
        type=str,
        choices=["zscore", "raw"],
        default="zscore",
        help=(
            "risk-budget score geometry: zscore removes cross-sectional scale; "
            "raw preserves causal return/volatility scale for temperature-aware allocation"
        ),
    )
    parser.add_argument(
        "--risk_score_epsilon",
        type=float,
        default=1e-4,
        help="causal volatility floor added to raw return/volatility risk scores",
    )
    parser.add_argument(
        "--risk_multiscale_residual_weight",
        type=float,
        default=1.0,
        help=(
            "strength of the learned multi-scale correction around the shortest "
            "causal risk scale"
        ),
    )
    parser.add_argument(
        "--risk_defensive_gate_floor",
        type=float,
        default=0.0,
        help="fixed causal floor for the learned downside-risk gate",
    )
    parser.add_argument(
        "--risk_momentum_short_weight",
        type=float,
        default=0.0,
        help="optional short-scale risk-momentum blend weight",
    )
    parser.add_argument(
        "--risk_momentum_residual_weight",
        type=float,
        default=0.0,
        help="learned Transformer allocation residual added to risk prior",
    )
    parser.add_argument(
        "--risk_gate_logit_scale",
        type=float,
        default=1.0,
        help=(
            "scale for learned risk-route gate logits; values below one keep "
            "the causal prior dominant and reduce seed sensitivity"
        ),
    )
    parser.add_argument(
        "--risk_gate_init",
        type=str,
        default="neutral",
        choices=["neutral", "decision"],
        help="deterministic initialization of the learned multi-route risk gates",
    )
    parser.add_argument(
        "--risk_forecast_weight",
        type=float,
        default=0.0,
        help="weight of the gated standardized return forecast in risk-budget logits",
    )
    parser.add_argument(
        "--risk_contrarian_weight",
        type=float,
        default=0.35,
        help="strength of the learned short-horizon reversal decision route",
    )
    parser.add_argument(
        "--risk_defensive_weight",
        type=float,
        default=0.15,
        help="strength of the learned downside-risk decision route",
    )
    parser.add_argument(
        "--risk_prior_bias",
        type=float,
        default=0.4,
        help="initial logit of the learned risk-prior gate",
    )
    parser.add_argument("--risk_downside_weight", type=float, default=0.25)
    parser.add_argument("--risk_drawdown_weight", type=float, default=0.10)
    parser.add_argument(
        "--risk_smoothing_temperature", type=float, default=1e-2
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--lradj",
        type=str,
        default="type1",
        choices=("type1", "type2", "type3"),
        help="type3: 10%% linear warmup, then cosine decay to 0.1x initial LR",
    )
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="L2 regularization applied by Adam to policy parameters",
    )
    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        choices=("objective", "sharpe"),
        default="objective",
        help=(
            "validation criterion used for checkpoint selection; objective "
            "uses the same end-to-end portfolio loss as training"
        ),
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help=(
            "optional exponential moving average of policy parameters used "
            "for validation/checkpointing; 0 disables it"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="test batch size; defaults to --batch_size (forced to 1 with --sequential_state)",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=parse_seed_list,
        default=[2023],
        help="integer seed or comma-separated seeds, e.g. 2023,2024,2025",
    )

    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", type=str, default="0,1")
    return parser.parse_args()


def main():
    args = parse_args()
    # Normalize this dataset-dependent structural-probe constraint before the
    # experiment key is constructed, so checkpoints record the effective cap.
    ensure_feasible_probe_upper_bound(args)
    log_path = setup_logger(args.log_dir, args.model_id)
    seed_values = list(args.seed)
    args.seed_list = seed_values
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(item) for item in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    print("\n" + "=" * 88)
    print(f"Seeds: {seed_values}")
    print(f"Log file: {log_path}")
    print("Full parameter configuration:")
    print(pformat(vars(args), sort_dicts=True, width=100))
    print("=" * 88)

    for run_index, seed in enumerate(seed_values):
        set_random_seed(seed)
        args.seed = seed
        if args.protocol == "sit":
            train_frequency = val_frequency = test_frequency = 1
            protocol_frequency = 1
        else:
            train_frequency = (
                args.train_rebalance_frequency
                if args.train_rebalance_frequency is not None
                else args.rebalance_frequency
            )
            val_frequency = (
                args.val_rebalance_frequency
                if args.val_rebalance_frequency is not None
                else args.rebalance_frequency
            )
            test_frequency = (
                args.test_rebalance_frequency
                if args.test_rebalance_frequency is not None
                else args.rebalance_frequency
            )
            protocol_frequency = args.rebalance_frequency
        setting = (
            f"{args.model_id}_KKTFormer-v0_{args.protocol}_dp{args.data_pool}"
            f"_ik{args.input_kind}"
            f"_w{args.window_size}_h{args.horizon}"
            f"_rb{protocol_frequency}"
            f"_trb{train_frequency}_vrb{val_frequency}_teb{test_frequency}"
            f"_lre{args.log_return_embed_dim}_de{args.date_embed_dim}"
            f"_ae{args.asset_embed_dim}_aes{args.asset_embedding_scale:g}_dm{args.d_model}"
            f"_aei{args.asset_embedding_init}_ph{args.portfolio_heads}"
            f"_mis{args.model_init_seed}"
            f"_pe{args.policy_experts}"
            f"_pha{args.portfolio_aggregation}"
            f"_phc{args.policy_head_consistency_weight:g}"
            f"_spf{args.spectral_policy_filters}_sph{args.spectral_policy_hidden}"
            f"_sps{args.spectral_policy_scale:g}"
            f"_tpf{args.tail_policy_filters}_tph{args.tail_policy_hidden}"
            f"_tps{args.tail_policy_scale:g}"
            f"_tpw{args.tail_policy_windows.replace(',', '-')}"
            f"_opb{args.ordered_policy_bins}"
            f"_ops{args.ordered_policy_scale:g}"
            f"_opw{args.ordered_policy_windows.replace(',', '-')}"
            f"_prs{args.policy_refinement_steps}"
            f"_prc{args.policy_refinement_scale:g}"
            f"_prw{args.policy_refinement_window}"
            f"_prr{args.policy_refinement_risk}"
            f"_prt{args.policy_refinement_tail_temperature:g}"
            f"_lgs{args.lpm_geometry_scale:g}"
            f"_lgw{args.lpm_geometry_window}"
            f"_lgi{args.lpm_geometry_init:g}"
            f"_lge{args.lpm_geometry_epsilon:g}"
            f"_ras{args.relation_attention_scale:g}"
            f"_rah{args.relation_attention_hidden}"
            f"_apb{args.use_asset_policy_bias}"
            f"_apbs{args.asset_policy_bias_scale:g}"
            f"_nh{args.n_heads}_nl{args.num_layers}"
            f"_oi{args.optimizer_iterations}_fb{args.feedback_mode}"
            f"_dl{args.decision_layer}_tp{args.temperature:g}"
            f"_saw{args.simplex_anchor_weight:g}"
            f"_maw{args.momentum_anchor_weight:g}"
            f"_mal{args.momentum_anchor_lookback}"
            f"_mat{args.momentum_anchor_temperature:g}"
            f"_daw{args.downside_anchor_weight:g}"
            f"_dal{args.downside_anchor_lookback}"
            f"_dap{args.downside_anchor_power:g}"
            f"_dae{args.downside_anchor_epsilon:g}"
            f"_rml{args.risk_momentum_lookback}_rms{args.risk_momentum_short_weight:g}"
            f"_rmr{args.risk_momentum_residual_weight:g}"
            f"_rgs{args.risk_gate_logit_scale:g}"
            f"_rgi{args.risk_gate_init}"
            f"_ema{args.ema_decay:g}"
            f"_cm{args.checkpoint_metric}"
            f"_wd{args.weight_decay:g}"
            f"_rfw{args.risk_forecast_weight:g}_fpw{args.forecast_weight:g}"
            f"_rcw{args.risk_contrarian_weight:g}_rdw{args.risk_defensive_weight:g}"
            f"_rmsw{args.risk_scale_windows.replace(',', '-')}"
            f"_rsn{args.risk_score_normalization}"
            f"_rse{args.risk_score_epsilon:g}"
            f"_rta{args.risk_turnover_aversion:g}"
            f"_ts{args.turnover_smoothing:g}"
            f"_mrw{args.mean_return_weight:g}"
            f"_sow{args.sortino_weight:g}"
            f"_shw{args.sharpe_weight:g}"
            f"_pss{args.portfolio_statistic_scope}"
            f"_cr{args.covariance_robustness:g}_cd{args.covariance_decay:g}"
            f"_krs{args.kkt_risk_scale:g}"
            f"_plb{args.probe_lower_bound:g}_pub{args.probe_upper_bound:g}"
            f"_sn{args.signal_normalization}_ss{args.signal_scale:g}"
            f"_lm{args.loss_mode}_cv{args.cvar_variant}"
            f"_ca{args.cvar_alpha:g}_ct{args.cvar_temperature:g}"
            f"_rw{args.regret_weight:g}_kw{args.ktr_weight:g}"
            f"_ka{args.ktr_tail_alpha:g}_kp{args.ktr_pressure_scale:g}"
            f"_pw{args.prediction_weight}"
            f"_seq{int(args.sequential_state)}_seed{seed}"
        )
        if args.entropy_regularization != 0.0:
            setting = setting.replace(
                f"_pw{args.prediction_weight}_seq",
                f"_pw{args.prediction_weight}_er{args.entropy_regularization:g}"
                f"_ee{args.entropy_epsilon:g}_seq",
            )
        full_setting = setting
        setting = filesystem_safe_setting(full_setting)
        if setting != full_setting:
            print(
                f"[Setting] compacted {len(full_setting.encode('utf-8'))}-byte "
                f"directory name to {len(setting.encode('utf-8'))} bytes: {setting}"
            )
        print("\n" + "=" * 88)
        print(f"Experiment {run_index + 1}/{len(seed_values)} (seed={seed})")
        print(f"Setting: {setting}")
        print(f">>>>>>> start training: {setting} >>>>>>>>>")
        experiment = EXP_KKT(args)
        experiment.train(setting)
        print(f">>>>>>> testing: {setting} >>>>>>>>>")
        experiment.eval(setting, load=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
