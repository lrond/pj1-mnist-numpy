from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = []
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocities.append({key: np.zeros_like(value) for key, value in layer.params.items()})
            else:
                self.velocities.append(None)
    
    def step(self):
        for layer, velocity in zip(self.model.layers, self.velocities):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    velocity[key] = self.mu * velocity[key] + layer.grads[key]
                    layer.params[key] -= self.init_lr * velocity[key]
