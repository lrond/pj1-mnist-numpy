import argparse
from pathlib import Path
from struct import unpack
import gzip

import numpy as np

import mynn as nn


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


def load_model(model_name, model_path):
    if model_name == "mlp":
        model = nn.models.Model_MLP()
    elif model_name == "cnn":
        model = nn.models.Model_CNN()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.load_model(model_path)
    return model


def predict_in_batches(model, images, batch_size):
    logits = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start:start + batch_size]
        logits.append(model(batch))
    return np.concatenate(logits, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST model.")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--test-images", type=Path, default=MNIST_DIR / "t10k-images-idx3-ubyte.gz")
    parser.add_argument("--test-labels", type=Path, default=MNIST_DIR / "t10k-labels-idx1-ubyte.gz")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    if model_path is None:
        model_path = BASE_DIR / "best_models" / args.model / "best_model.pickle"

    model = load_model(args.model, model_path)
    test_imgs = read_images(args.test_images)
    test_labs = read_labels(args.test_labels)

    logits = predict_in_batches(model, test_imgs, args.batch_size)
    print(f"test accuracy: {nn.metric.accuracy(logits, test_labs):.5f}")
