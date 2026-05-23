from abc import abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "Cannot call backward before forward."
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.b = initialize_method(size=(1, out_channels))
        self.grads = {'W' : None, 'b' : None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.input = None
        self.padded_input = None
        self.windows = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        if X.ndim != 4:
            raise ValueError("conv2D expects input with shape [batch, channels, H, W].")
        if X.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {X.shape[1]}.")

        self.input = X
        if self.padding > 0:
            self.padded_input = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            self.padded_input = X

        k_h, k_w = self.kernel_size
        if self.padded_input.shape[2] < k_h or self.padded_input.shape[3] < k_w:
            raise ValueError("Kernel size is larger than the padded input.")

        all_windows = sliding_window_view(self.padded_input, (k_h, k_w), axis=(2, 3))
        self.windows = all_windows[:, :, ::self.stride, ::self.stride, :, :]
        output = np.einsum('n c h w r s, o c r s -> n o h w', self.windows, self.W)
        output += self.b.reshape(1, self.out_channels, 1, 1)
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None and self.windows is not None, "Cannot call backward before forward."
        if grads.shape[1] != self.out_channels:
            raise ValueError(f"Expected {self.out_channels} output-channel gradients, got {grads.shape[1]}.")

        self.grads['W'] = np.einsum('n o h w, n c h w r s -> o c r s', grads, self.windows)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=False).reshape(1, self.out_channels)

        d_padded = np.zeros_like(self.padded_input)
        k_h, k_w = self.kernel_size
        out_h, out_w = grads.shape[2], grads.shape[3]
        for r in range(k_h):
            h_slice = slice(r, r + self.stride * out_h, self.stride)
            for s in range(k_w):
                w_slice = slice(s, s + self.stride * out_w, self.stride)
                d_padded[:, :, h_slice, w_slice] += np.einsum(
                    'n o h w, o c -> n c h w',
                    grads,
                    self.W[:, :, r, s],
                )

        if self.padding > 0:
            return d_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return d_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    """
    Flatten image-like tensors to [batch_size, features].
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        assert self.input_shape is not None, "Cannot call backward before forward."
        return grads.reshape(self.input_shape)

class MaxPool2D(Layer):
    """
    Max pooling layer for image-like tensors with shape [batch, channels, H, W].
    """
    def __init__(self, pool_size=2, stride=None) -> None:
        super().__init__()
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
        self.input = None
        self.windows = None
        self.max_values = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim != 4:
            raise ValueError("MaxPool2D expects input with shape [batch, channels, H, W].")
        self.input = X
        p_h, p_w = self.pool_size
        if X.shape[2] < p_h or X.shape[3] < p_w:
            raise ValueError("Pool size is larger than the input.")
        all_windows = sliding_window_view(X, (p_h, p_w), axis=(2, 3))
        self.windows = all_windows[:, :, ::self.stride, ::self.stride, :, :]
        self.max_values = np.max(self.windows, axis=(-1, -2), keepdims=True)
        return self.max_values[..., 0, 0]

    def backward(self, grads):
        assert self.input is not None and self.windows is not None, "Cannot call backward before forward."
        p_h, p_w = self.pool_size
        out_h, out_w = grads.shape[2], grads.shape[3]
        d_input = np.zeros_like(self.input)
        mask = self.windows == self.max_values
        tie_counts = np.sum(mask, axis=(-1, -2), keepdims=True)
        distributed = mask * (grads[..., None, None] / tie_counts)
        for r in range(p_h):
            h_slice = slice(r, r + self.stride * out_h, self.stride)
            for s in range(p_w):
                w_slice = slice(s, s + self.stride * out_w, self.stride)
                d_input[:, :, h_slice, w_slice] += distributed[..., r, s]
        return d_input

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.labels = None
        self.predicts = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        labels = labels.astype(np.int64)
        if predicts.ndim != 2:
            raise ValueError("predicts must have shape [batch_size, num_classes].")
        if predicts.shape[0] != labels.shape[0]:
            raise ValueError("predicts and labels must have the same batch size.")

        self.labels = labels
        self.predicts = predicts
        self.probs = softmax(predicts) if self.has_softmax else predicts

        batch_indices = np.arange(labels.shape[0])
        eps = 1e-12
        return -np.mean(np.log(self.probs[batch_indices, labels] + eps))
    
    def backward(self):
        # first compute the grads from the loss to the input
        assert self.probs is not None and self.labels is not None, "Cannot call backward before forward."
        batch_size = self.labels.shape[0]
        batch_indices = np.arange(batch_size)
        eps = 1e-12

        if self.has_softmax:
            self.grads = self.probs.copy()
            self.grads[batch_indices, self.labels] -= 1.0
            self.grads /= batch_size
        else:
            self.grads = np.zeros_like(self.probs)
            self.grads[batch_indices, self.labels] = -1.0 / (self.probs[batch_indices, self.labels] + eps)
            self.grads /= batch_size

        # Then send the grads to model for back propagation
        if self.model is not None:
            self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
