### PJ1 MNIST NumPy Framework

This folder contains a runnable NumPy implementation for the required MLP and CNN MNIST experiments.

### Implemented Components

1. `mynn/op.py`
   - `Linear.forward` and `Linear.backward`
   - `MultiCrossEntropyLoss` with softmax
   - `conv2D.forward` and `conv2D.backward`
   - `Flatten`
   - `MaxPool2D`
2. `mynn/models.py`
   - `Model_MLP`
   - `Model_CNN` with convolution, ReLU, max pooling, and fully connected layers
3. `mynn/optimizer.py`
   - `SGD`
   - `MomentGD`
4. `mynn/lr_scheduler.py`
   - `StepLR`
   - `MultiStepLR`
   - `ExponentialLR`

### Quick Verification

From the project root:

```bash
python3 -m pytest codes/test_core.py -q
```

### Train Models

From the project root:

```bash
python3 codes/test_train.py --model mlp --epochs 5
python3 codes/test_train.py --model cnn --epochs 5 --optimizer momentum --lr 0.02 --run-name cnn_fair_momentum
```

For a fast smoke test:

```bash
python3 codes/test_train.py --model mlp --epochs 1 --train-limit 512 --dev-size 128 --batch-size 64 --eval-interval 2 --log-iters 2
python3 codes/test_train.py --model cnn --epochs 1 --train-limit 128 --dev-size 64 --batch-size 16 --eval-interval 2 --log-iters 2
```

Saved models and learning curves are written to:

```text
codes/best_models/<model_name>/best_model.pickle
codes/best_models/<model_name>/learning_curve.png
```

### Evaluate Saved Models

```bash
python3 codes/test_model.py --model mlp
python3 codes/test_model.py --model cnn
```

### Part C: Optimization Comparison

To isolate momentum, compare SGD and Momentum under the same learning rate and scheduler:

```bash
python3 codes/test_train.py --model mlp --epochs 5 --optimizer sgd --scheduler multistep --lr 0.02 --run-name mlp_sgd_lr002
python3 codes/test_train.py --model mlp --epochs 5 --optimizer momentum --scheduler multistep --lr 0.02 --run-name mlp_fair_momentum
```

### Part C: Error Analysis and Visualization

After training a saved model:

```bash
python3 codes/weight_visualization.py --model mlp
python3 codes/weight_visualization.py --model cnn
```

Figures are saved to:

```text
codes/analysis_outputs/
```

Use the generated learning curves, confusion matrix, misclassified examples, and weight/kernel visualizations in the report.
