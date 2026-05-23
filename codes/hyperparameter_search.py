import argparse
from copy import deepcopy

import numpy as np

from test_train import train_one


def run_optimization_comparison(base_args):
    settings = [
        {
            "name": "sgd",
            "optimizer": "sgd",
            "scheduler": "none",
            "lr": base_args.lr,
        },
        {
            "name": "sgd_multistep",
            "optimizer": "sgd",
            "scheduler": "multistep",
            "lr": base_args.lr,
        },
        {
            "name": "momentum_multistep",
            "optimizer": "momentum",
            "scheduler": "multistep",
            "lr": base_args.lr,
        },
    ]

    results = []
    for setting in settings:
        args = deepcopy(base_args)
        args.optimizer = setting["optimizer"]
        args.scheduler = setting["scheduler"]
        args.lr = setting["lr"]
        args.run_name = f"{args.model}_{setting['name']}"
        print(f"\n=== Optimization experiment: {setting['name']} ===")
        score = train_one(args.model, args)
        results.append((setting["name"], score))

    print("\nOptimization comparison")
    for name, score in results:
        print(f"{name}: best validation accuracy = {score:.5f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run focused Part C optimization comparisons.")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--milestones", type=int, nargs="*", default=[400, 800])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--dev-size", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--log-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=309)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    run_optimization_comparison(args)
