import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from test_model import BASE_DIR, load_model, predict_in_batches, read_images, read_labels


def confusion_matrix(labels, preds, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[label, pred] += 1
    return matrix


def save_confusion_matrix(matrix, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_misclassified_examples(images, labels, preds, output_path, max_examples=16):
    wrong = np.where(labels != preds)[0][:max_examples]
    if wrong.size == 0:
        print("No misclassified examples found.")
        return

    cols = 4
    rows = int(np.ceil(wrong.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, index in zip(axes, wrong):
        ax.imshow(images[index].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{labels[index]} P:{preds[index]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_mlp_weights(model, output_path, max_units=36):
    first_linear = next(layer for layer in model.layers if layer.optimizable)
    weights = first_linear.params["W"]
    count = min(max_units, weights.shape[1])
    cols = 6
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i in range(count):
        axes[i].imshow(weights[:, i].reshape(28, 28), cmap="coolwarm")
        axes[i].set_title(str(i))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_cnn_kernels(model, output_path):
    first_conv = next(layer for layer in model.layers if layer.__class__.__name__ == "conv2D")
    kernels = first_conv.params["W"]
    count = kernels.shape[0]
    cols = min(8, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i in range(count):
        kernel = kernels[i, 0]
        axes[i].imshow(kernel, cmap="coolwarm")
        axes[i].set_title(str(i))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Part C error-analysis and visualization figures.")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "analysis_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path
    if model_path is None:
        model_path = BASE_DIR / "best_models" / args.model / "best_model.pickle"

    model = load_model(args.model, model_path)
    images = read_images(BASE_DIR / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz")
    labels = read_labels(BASE_DIR / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz")
    if args.limit is not None:
        images = images[:args.limit]
        labels = labels[:args.limit]

    logits = predict_in_batches(model, images, args.batch_size)
    preds = np.argmax(logits, axis=1)
    matrix = confusion_matrix(labels, preds)

    save_confusion_matrix(matrix, args.output_dir / f"{args.model}_confusion_matrix.png")
    save_misclassified_examples(images, labels, preds, args.output_dir / f"{args.model}_misclassified.png")
    if args.model == "mlp":
        save_mlp_weights(model, args.output_dir / "mlp_first_layer_weights.png")
    else:
        save_cnn_kernels(model, args.output_dir / "cnn_kernels.png")

    print(f"accuracy: {(preds == labels).mean():.5f}")
    print(f"saved figures to: {args.output_dir}")
