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
    parser.add_argument("--data_pool", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--rebalance_frequency", type=int, default=20)
    parser.add_argument("--upper_bound", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--covariance_epsilon", type=float, default=1e-6)
    parser.add_argument("--transaction_cost_bps", type=float, default=0.0)

    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--optimizer_iterations", type=int, default=100)
    parser.add_argument("--projection_iterations", type=int, default=64)
    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["prediction", "utility", "cvar", "regret", "hybrid"],
        default="prediction",
        help="stage-5 training objective",
    )
    parser.add_argument("--prediction_loss", type=str, default="MSE")
    parser.add_argument("--cvar_alpha", type=float, default=0.95)
    parser.add_argument("--cvar_temperature", type=float, default=1e-3)
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
        setting = (
            f"{args.model_id}_KKTFormer-v0_dp{args.data_pool}"
            f"_w{args.window_size}_h{args.horizon}"
            f"_rb{args.rebalance_frequency}_dm{args.d_model}"
            f"_nh{args.n_heads}_nl{args.num_layers}"
            f"_oi{args.optimizer_iterations}_lm{args.loss_mode}"
            f"_pw{args.prediction_weight}_{iteration}"
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
