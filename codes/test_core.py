import os
import tempfile

import numpy as np

import mynn as nn


def _numeric_gradient(param, objective, eps=1e-5):
    grad = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]
        param[idx] = old_value + eps
        plus = objective()
        param[idx] = old_value - eps
        minus = objective()
        param[idx] = old_value
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def test_linear_forward_backward_matches_manual_values():
    layer = nn.op.Linear(2, 3, initialize_method=lambda size: np.zeros(size))
    layer.W[...] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer.b[...] = np.array([[0.5, -0.5, 1.0]])

    X = np.array([[1.0, 2.0], [-1.0, 0.5]])
    out = layer(X)
    expected_out = X @ layer.W + layer.b
    assert np.allclose(out, expected_out)

    grad = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    grad_input = layer.backward(grad)

    assert np.allclose(layer.grads["W"], X.T @ grad)
    assert np.allclose(layer.grads["b"], np.sum(grad, axis=0, keepdims=True))
    assert np.allclose(grad_input, grad @ layer.W.T)


def test_linear_backward_matches_finite_difference():
    layer = nn.op.Linear(3, 2, initialize_method=lambda size: np.zeros(size))
    layer.W[...] = np.array([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.7]], dtype=np.float64)
    layer.b[...] = np.array([[0.05, -0.02]], dtype=np.float64)
    X = np.array([[0.3, -0.7, 1.1], [-0.4, 0.2, 0.5]], dtype=np.float64)
    upstream = np.array([[0.6, -0.3], [0.2, 0.4]], dtype=np.float64)

    def objective():
        return np.sum(layer(X) * upstream)

    layer(X)
    grad_input = layer.backward(upstream)

    num_w = _numeric_gradient(layer.W, objective)
    num_b = _numeric_gradient(layer.b, objective)
    num_x = _numeric_gradient(X, objective)

    assert np.allclose(layer.grads["W"], num_w, atol=1e-6)
    assert np.allclose(layer.grads["b"], num_b, atol=1e-6)
    assert np.allclose(grad_input, num_x, atol=1e-6)


def test_cross_entropy_loss_and_backward_gradient():
    loss_fn = nn.op.MultiCrossEntropyLoss(model=None, max_classes=2)
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 1])

    loss = loss_fn(logits, labels)
    probs = nn.op.softmax(logits)
    expected_loss = -np.mean(np.log(probs[np.arange(2), labels]))
    assert np.allclose(loss, expected_loss)

    loss_fn.backward()
    expected_grad = probs.copy()
    expected_grad[np.arange(2), labels] -= 1.0
    expected_grad /= 2
    assert np.allclose(loss_fn.grads, expected_grad)


def test_cross_entropy_backward_matches_finite_difference():
    loss_fn = nn.op.MultiCrossEntropyLoss(model=None, max_classes=3)
    logits = np.array([[0.7, -1.2, 0.3], [-0.4, 0.9, 0.1]], dtype=np.float64)
    labels = np.array([2, 1])

    loss_fn(logits, labels)
    analytic = loss_fn.backward().copy()

    def objective():
        return loss_fn(logits, labels)

    numeric = _numeric_gradient(logits, objective)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_conv2d_forward_backward_shapes_and_values():
    conv = nn.op.conv2D(1, 1, 2, stride=1, padding=0, initialize_method=lambda size: np.zeros(size))
    conv.W[...] = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    conv.b[...] = np.array([[0.5]])

    X = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    out = conv(X)
    expected = np.array([[[[-3.5, -3.5], [-3.5, -3.5]]]])
    assert np.allclose(out, expected)

    grad_input = conv.backward(np.ones_like(out))
    assert grad_input.shape == X.shape
    assert conv.grads["W"].shape == conv.W.shape
    assert conv.grads["b"].shape == conv.b.shape
    assert np.allclose(conv.grads["W"], np.array([[[[12.0, 16.0], [24.0, 28.0]]]]))
    assert np.allclose(conv.grads["b"], np.array([[4.0]]))


def test_conv2d_backward_matches_finite_difference():
    conv = nn.op.conv2D(1, 2, 2, stride=1, padding=1, initialize_method=lambda size: np.zeros(size))
    conv.W[...] = np.array(
        [[[[0.2, -0.1], [0.4, 0.3]]], [[[-0.3, 0.5], [0.1, -0.2]]]],
        dtype=np.float64,
    )
    conv.b[...] = np.array([[0.05, -0.07]], dtype=np.float64)
    X = np.array([[[[0.2, -0.4, 0.1], [0.7, -0.3, 0.5], [-0.6, 0.8, -0.2]]]], dtype=np.float64)
    upstream = np.linspace(-0.3, 0.4, num=2 * 4 * 4, dtype=np.float64).reshape(1, 2, 4, 4)

    def objective():
        return np.sum(conv(X) * upstream)

    conv(X)
    grad_input = conv.backward(upstream)

    num_w = _numeric_gradient(conv.W, objective)
    num_b = _numeric_gradient(conv.b, objective)
    num_x = _numeric_gradient(X, objective)

    assert np.allclose(conv.grads["W"], num_w, atol=1e-6)
    assert np.allclose(conv.grads["b"], num_b, atol=1e-6)
    assert np.allclose(grad_input, num_x, atol=1e-6)


def test_maxpool2d_forward_backward_routes_to_maxima():
    pool = nn.op.MaxPool2D(pool_size=2)
    X = np.array([[[[1.0, 5.0], [3.0, 2.0]]]])
    out = pool(X)
    assert np.allclose(out, np.array([[[[5.0]]]]))

    grad_input = pool.backward(np.ones_like(out))
    assert np.allclose(grad_input, np.array([[[[0.0, 1.0], [0.0, 0.0]]]]))


def test_cnn_can_save_load_and_predict():
    model = nn.models.Model_CNN(input_shape=(1, 4, 4), conv_channels=2, kernel_size=3, hidden_dim=5)
    X = np.random.default_rng(0).normal(size=(3, 1, 4, 4))
    logits = model(X)
    assert logits.shape == (3, 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cnn.pickle")
        model.save_model(path)
        loaded = nn.models.Model_CNN()
        loaded.load_model(path)
        assert np.allclose(model(X), loaded(X))


def test_sgd_updates_layer_arrays_in_place():
    model = nn.models.Model_MLP([2, 1], "ReLU", [0.0])
    layer = model.layers[0]
    layer.W[...] = np.array([[1.0], [2.0]])
    layer.b[...] = np.array([[0.5]])
    layer.grads["W"] = np.array([[0.1], [0.2]])
    layer.grads["b"] = np.array([[0.3]])

    optimizer = nn.optimizer.SGD(init_lr=0.5, model=model)
    optimizer.step()

    assert np.allclose(layer.W, np.array([[0.95], [1.9]]))
    assert np.allclose(layer.b, np.array([[0.35]]))
    assert layer.params["W"] is layer.W
    assert layer.params["b"] is layer.b


def test_momentum_and_multistep_lr_change_updates():
    model = nn.models.Model_MLP([1, 1], "ReLU", [0.0])
    layer = model.layers[0]
    layer.W[...] = np.array([[1.0]])
    layer.b[...] = np.array([[0.0]])
    layer.grads["W"] = np.array([[1.0]])
    layer.grads["b"] = np.array([[0.0]])

    optimizer = nn.optimizer.MomentGD(init_lr=0.1, model=model, mu=0.9)
    optimizer.step()
    optimizer.step()
    assert np.allclose(layer.W, np.array([[0.71]]))

    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2], gamma=0.5)
    scheduler.step()
    assert np.allclose(optimizer.init_lr, 0.1)
    scheduler.step()
    assert np.allclose(optimizer.init_lr, 0.05)
