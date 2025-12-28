"""
Activation functions for neural networks.
includes ReLU, Sigmoid, Tanh, and Linear.
"""

import numpy as np
from base import Module


class Activation(Module):
    """Base class for activation functions"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    def _formula(self, x: np.ndarray):
        """Activation function formula"""
        raise NotImplementedError("Activation functions must implement the _formula method.")
    def _gradient(self, x: np.ndarray):
        """Gradient of the activation function"""
        raise NotImplementedError("Activation functions must implement the _gradient method.")
    def forward(self, x):
        if self.training:
            self.cache = x
        return self._formula(x)
    def backward(self, grad_output):
        return grad_output * self._gradient(self.cache)

class ReLU(Activation):
    """Rectified Linear Unit activation function
    f(x) = max(0, x)
    """
    def _formula(self, x):
        return np.maximum(0, x)
    def _gradient(self, x):
        return (x > 0).astype(x.dtype)

class Sigmoid(Activation):
    """Sigmoid activation function
    f(x) = 1 / (1 + exp(-x))
    """
    def _formula(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    def _gradient(self, x):
        return self._formula(x) * (1 - self._formula(x))

class Tanh(Activation):
    """Hyperbolic Tangent activation function
    f(x) = tanh(x)
    """
    def _formula(self, x):
        return np.tanh(x)
    def _gradient(self, x):
        return 1 - np.tanh(x) ** 2

class Linear(Activation):
    """Linear activation function
    f(x) = x
    """
    def _formula(self, x):
        return x
    def _gradient(self, x):
        return np.ones_like(x)


def testing():
    import torch
    from torch.nn import (
        ReLU as TorchReLU,
        Sigmoid as TorchSigmoid,
        Tanh as TorchTanh,
        Identity as TorchLinear,
    )
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    relu = ReLU()
    print("ReLU Forward:", relu.forward(x))
    print("ReLU Torch Forward:", TorchReLU()(torch.from_numpy(x)))
    # relu.training = True
    relu.backward(np.ones_like(x))
    print("ReLU Backward:", relu.backward(np.ones_like(x)))
    
    sigmoid = Sigmoid()
    print("\nSigmoid Forward:", sigmoid.forward(x))
    print("Sigmoid Torch Forward:", TorchSigmoid()(torch.from_numpy(x)))
    # sigmoid.training = True
    sigmoid.backward(np.ones_like(x))
    print("Sigmoid Backward:", sigmoid.backward(np.ones_like(x)))
    
    tanh = Tanh()
    print("\nTanh Forward:", tanh.forward(x))
    print("Tanh Torch Forward:", TorchTanh()(torch.from_numpy(x)))
    # tanh.training = True
    tanh.backward(np.ones_like(x))
    print("Tanh Backward:", tanh.backward(np.ones_like(x)))
    
    linear = Linear()
    print("\nLinear Forward:", linear.forward(x))
    print("Linear Torch Forward:", TorchLinear()(torch.from_numpy(x)))
    #linear.training = True
    linear.backward(np.ones_like(x))
    print("Linear Backward:", linear.backward(np.ones_like(x)))
    
if __name__ == "__main__":
    testing()