# PJ1 MNIST NumPy Neural Network

This repository contains the code for Project 1 of Neural Network and Deep Learning.

Implemented from scratch with NumPy:

- linear layer forward/backward
- softmax cross-entropy loss
- 2D convolution forward/backward
- max pooling forward/backward
- MLP and CNN models
- SGD, momentum gradient descent, and learning-rate schedulers
- training, testing, and visualization scripts

The MNIST dataset and trained model weights are intentionally not tracked in Git. Put the provided MNIST files under `codes/dataset/MNIST/` before running the experiments.

Trained checkpoints are available on ModelScope:
https://www.modelscope.cn/models/lrond1/pj1-mnist-numpy-checkpoints

## Quick Check

```bash
python3 -m pytest codes/test_core.py -q
```

## Main Experiments

```bash
python3 codes/test_train.py --model mlp --epochs 5 --run-name mlp_baseline
python3 codes/test_train.py --model mlp --epochs 5 --optimizer momentum --scheduler multistep --lr 0.02 --run-name mlp_fair_momentum
python3 codes/test_train.py --model cnn --epochs 5 --optimizer momentum --scheduler multistep --lr 0.02 --run-name cnn_fair_momentum
```

## Evaluation

```bash
python3 codes/test_model.py --model mlp --model-path codes/best_models/mlp_baseline/best_model.pickle
python3 codes/test_model.py --model mlp --model-path codes/best_models/mlp_fair_momentum/best_model.pickle
python3 codes/test_model.py --model cnn --model-path codes/best_models/cnn_fair_momentum/best_model.pickle
```

## Report

The LaTeX report and Overleaf-ready zip are generated under `report/`.
