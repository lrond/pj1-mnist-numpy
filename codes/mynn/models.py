from .op import *
import pickle
import numpy as np


def _scaled_normal(scale):
    def initialize(size):
        return np.random.normal(0.0, scale, size=size)
    return initialize


def _assign_layer_params(layer, saved_layer):
    layer.W = saved_layer['W']
    layer.b = saved_layer['b']
    layer.params['W'] = layer.W
    layer.params['b'] = layer.b
    layer.weight_decay = saved_layer.get('weight_decay', False)
    layer.weight_decay_lambda = saved_layer.get('lambda', 1e-8)

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(
                    in_dim=size_list[i],
                    out_dim=size_list[i + 1],
                    initialize_method=_scaled_normal(np.sqrt(2.0 / size_list[i])),
                )
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        if isinstance(param_list, dict):
            if param_list.get('model_type') != 'Model_MLP':
                raise ValueError(f"Expected Model_MLP parameters, got {param_list.get('model_type')}.")
            self.size_list = param_list['size_list']
            self.act_func = param_list['act_func']
            saved_layers = param_list['layers']
        else:
            self.size_list = param_list[0]
            self.act_func = param_list[1]
            saved_layers = param_list[2:]

        self.layers = []
        optimizable_index = 0
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            _assign_layer_params(layer, saved_layers[optimizable_index])
            optimizable_index += 1
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(
        self,
        input_shape=(1, 28, 28),
        num_classes=10,
        conv_channels=8,
        kernel_size=3,
        pool_size=2,
        hidden_dim=128,
        act_func='ReLU',
        lambda_list=None,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.act_func = act_func
        self.layers = []

        if self.input_shape is not None:
            self._build(lambda_list=lambda_list)

    def _build(self, lambda_list=None):
        if self.act_func != 'ReLU':
            raise NotImplementedError("Model_CNN currently supports ReLU only.")
        in_channels, height, width = self.input_shape
        kernel = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
        conv_scale = np.sqrt(2.0 / (in_channels * kernel * kernel))
        conv = conv2D(
            in_channels=in_channels,
            out_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            initialize_method=_scaled_normal(conv_scale),
        )
        conv_out_h = height - kernel + 1
        conv_out_w = width - kernel + 1
        if conv_out_h <= 0 or conv_out_w <= 0:
            raise ValueError("CNN kernel size is too large for the input shape.")
        pool_out_h = (conv_out_h - self.pool_size) // self.pool_size + 1
        pool_out_w = (conv_out_w - self.pool_size) // self.pool_size + 1
        if pool_out_h <= 0 or pool_out_w <= 0:
            raise ValueError("CNN pool size is too large for the convolution output.")

        flat_dim = self.conv_channels * pool_out_h * pool_out_w
        fc1 = Linear(
            in_dim=flat_dim,
            out_dim=self.hidden_dim,
            initialize_method=_scaled_normal(np.sqrt(2.0 / flat_dim)),
        )
        fc2 = Linear(
            in_dim=self.hidden_dim,
            out_dim=self.num_classes,
            initialize_method=_scaled_normal(np.sqrt(2.0 / self.hidden_dim)),
        )

        optimizable_layers = [conv, fc1, fc2]
        if lambda_list is not None:
            for layer, weight_decay_lambda in zip(optimizable_layers, lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = weight_decay_lambda

        self.layers = [conv, ReLU(), MaxPool2D(pool_size=self.pool_size), Flatten(), fc1, ReLU(), fc2]

    def __call__(self, X):
        return self.forward(X)

    def _prepare_input(self, X):
        if X.ndim == 2:
            expected_features = np.prod(self.input_shape)
            if X.shape[1] != expected_features:
                raise ValueError(f"Expected {expected_features} flat features, got {X.shape[1]}.")
            return X.reshape(X.shape[0], *self.input_shape)
        if X.ndim == 3 and self.input_shape[0] == 1:
            return X[:, None, :, :]
        return X

    def forward(self, X):
        assert self.layers, 'Model has not initialized yet. Use model.load_model or create it with input_shape.'
        outputs = self._prepare_input(X)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        if not isinstance(param_list, dict) or param_list.get('model_type') != 'Model_CNN':
            raise ValueError("Expected a saved Model_CNN parameter dictionary.")

        self.input_shape = tuple(param_list['input_shape'])
        self.num_classes = param_list['num_classes']
        self.conv_channels = param_list['conv_channels']
        self.kernel_size = param_list['kernel_size']
        self.pool_size = param_list.get('pool_size', 2)
        self.hidden_dim = param_list['hidden_dim']
        self.act_func = param_list.get('act_func', 'ReLU')
        self._build()

        saved_layers = param_list['layers']
        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, saved_layer in zip(optimizable_layers, saved_layers):
            _assign_layer_params(layer, saved_layer)
        
    def save_model(self, save_path):
        param_list = {
            'model_type': 'Model_CNN',
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'conv_channels': self.conv_channels,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'hidden_dim': self.hidden_dim,
            'act_func': self.act_func,
            'layers': [],
        }
        for layer in self.layers:
            if layer.optimizable:
                param_list['layers'].append({
                    'W' : layer.params['W'],
                    'b' : layer.params['b'],
                    'weight_decay' : layer.weight_decay,
                    'lambda' : layer.weight_decay_lambda,
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
