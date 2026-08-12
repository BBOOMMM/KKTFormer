"""Command-line entry point for KKTFormer-v0."""

import argparse
import random

import numpy as np
import torch

from exp.exp_main_kkt import EXP_KKT


def parse_args():
    parser = argparse.ArgumentParser(description="Train KKTFormer-v0")
    parser.add_argument("--model_id", type=str, default="kkt_v0")
    parser.add_argument("--root_path", type=str, default="./asset_data/")
    parser.add_argument("--data_path", type=str, default="full_dataset.csv")
    parser.add_argument("--context_root", type=str, default="")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints_kkt/")
    parser.add_argument("--results_path", type=str, default="./results_kkt/")
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
    parser.add_argument("--budget_target", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--covariance_epsilon", type=float, default=1e-6)
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
        choices=["none", "dual", "jacobian"],
        default="dual",
        help="stage-6 optimizer feedback variant",
    )
    parser.add_argument("--active_tolerance", type=float, default=1e-5)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--optimizer_iterations",
        type=int,
        default=10,
        help="warm-started final optimizer steps; 10 is sufficient for the default QP",
    )
    parser.add_argument("--probe_optimizer_iterations", type=int, default=5)
    parser.add_argument("--projection_iterations", type=int, default=64)
    parser.add_argument("--constraint_projection_iterations", type=int, default=20)
    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["prediction", "utility", "cvar", "regret", "hybrid"],
        default="cvar",
        help="end-to-end realized-return objective (sequence KKTFormer requires cvar)",
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
    parser.add_argument("--kkt_bias_rank", type=int, default=4)
    parser.add_argument(
        "--prediction_weight",
        type=float,
        default=0.1,
        help="prediction-loss weight for hybrid regret training",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", type=str, default="0,1")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(item) for item in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    for iteration in range(args.itr):
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
            f"_w{args.window_size}_h{args.horizon}"
            f"_rb{protocol_frequency}"
            f"_trb{train_frequency}_vrb{val_frequency}_teb{test_frequency}"
            f"_dm{args.d_model}"
            f"_nh{args.n_heads}_nl{args.num_layers}"
            f"_oi{args.optimizer_iterations}_fb{args.feedback_mode}"
            f"_sn{args.signal_normalization}_ss{args.signal_scale:g}"
            f"_lm{args.loss_mode}_cv{args.cvar_variant}"
            f"_pw{args.prediction_weight}_seq{int(args.sequential_state)}_{iteration}"
        )
        if args.entropy_regularization != 0.0:
            setting = setting.replace(
                f"_pw{args.prediction_weight}_seq",
                f"_pw{args.prediction_weight}_er{args.entropy_regularization:g}"
                f"_ee{args.entropy_epsilon:g}_seq",
            )
        print(f">>>>>>> start training: {setting} >>>>>>>>>")
        experiment = EXP_KKT(args)
        experiment.train(setting)
        print(f">>>>>>> testing: {setting} >>>>>>>>>")
        experiment.eval(setting, load=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
