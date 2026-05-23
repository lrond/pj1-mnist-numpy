import argparse
from pathlib import Path
from struct import unpack
import gzip

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot


BASE_DIR = Path(__file__).resolve().parent
MNIST_DIR = BASE_DIR / "dataset" / "MNIST"


def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images.astype(np.float32) / 255.0


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_train_valid(dev_size=10000, train_limit=None, seed=309):
    train_imgs = read_images(MNIST_DIR / "train-images-idx3-ubyte.gz")
    train_labs = read_labels(MNIST_DIR / "train-labels-idx1-ubyte.gz")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(np.arange(train_imgs.shape[0]))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:dev_size]
    valid_labs = train_labs[:dev_size]
    train_imgs = train_imgs[dev_size:]
    train_labs = train_labs[dev_size:]

    if train_limit is not None:
        train_imgs = train_imgs[:train_limit]
        train_labs = train_labs[:train_limit]

    return (train_imgs, train_labs), (valid_imgs, valid_labs)


def build_model(model_name, weight_decay, mlp_hidden, cnn_conv_channels, cnn_hidden_dim):
    if model_name == "mlp":
        lambdas = [weight_decay, weight_decay] if weight_decay > 0 else None
        return nn.models.Model_MLP([28 * 28, mlp_hidden, 10], "ReLU", lambdas)
    if model_name == "cnn":
        lambdas = [weight_decay, weight_decay, weight_decay] if weight_decay > 0 else None
        return nn.models.Model_CNN(
            input_shape=(1, 28, 28),
            conv_channels=cnn_conv_channels,
            kernel_size=3,
            pool_size=2,
            hidden_dim=cnn_hidden_dim,
            lambda_list=lambdas,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_optimizer(name, init_lr, model, momentum):
    if name == "sgd":
        return nn.optimizer.SGD(init_lr=init_lr, model=model)
    if name == "momentum":
        return nn.optimizer.MomentGD(init_lr=init_lr, model=model, mu=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name, optimizer, milestones, gamma):
    if name == "none":
        return None
    if name == "multistep":
        return nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
    if name == "exponential":
        return nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    raise ValueError(f"Unknown scheduler: {name}")


def train_one(model_name, args):
    train_set, valid_set = load_train_valid(
        dev_size=args.dev_size,
        train_limit=args.train_limit,
        seed=args.seed,
    )
    model = build_model(model_name, args.weight_decay, args.mlp_hidden, args.cnn_conv_channels, args.cnn_hidden_dim)
    optimizer = build_optimizer(args.optimizer, args.lr, model, args.momentum)
    scheduler = build_scheduler(args.scheduler, optimizer, args.milestones, args.gamma)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
        scheduler=scheduler,
    )

    save_name = getattr(args, "run_name", None) or model_name
    save_dir = BASE_DIR / "best_models" / save_name
    runner.train(
        train_set,
        valid_set,
        num_epochs=args.epochs,
        log_iters=args.log_iters,
        eval_interval=args.eval_interval,
        save_dir=str(save_dir),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_tight_layout(True)
    plot(runner, axes)
    curve_path = save_dir / "learning_curve.png"
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    print(f"[{model_name}] best validation accuracy: {runner.best_score:.5f}")
    print(f"[{model_name}] saved model: {save_dir / 'best_model.pickle'}")
    print(f"[{model_name}] saved curve: {curve_path}")
    return runner.best_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP/CNN models on MNIST with the NumPy PJ1 framework.")
    parser.add_argument("--model", choices=["mlp", "cnn", "all"], default="mlp")
    parser.add_argument("--optimizer", choices=["sgd", "momentum"], default="sgd")
    parser.add_argument("--scheduler", choices=["none", "multistep", "exponential"], default="multistep")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--cnn-conv-channels", type=int, default=16)
    parser.add_argument("--cnn-hidden-dim", type=int, default=128)
    parser.add_argument("--milestones", type=int, nargs="*", default=[800, 2400, 4000])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--dev-size", type=int, default=10000)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--log-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=309)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    models = ["mlp", "cnn"] if args.model == "all" else [args.model]
    for model_name in models:
        train_one(model_name, args)
