"""
Activation functions for neural networks.
includes ReLU, Sigmoid, Tanh, and Linear.
"""

import numpy as np
from .base import Module
from typing import Any


class Activation(Module):
    """Base class for activation functions"""
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
    def _formula(self, x):
        return np.maximum(0, x)
    def _gradient(self, x):
        return (x > 0).astype(x.dtype)

class Sigmoid(Activation):
    """Sigmoid activation function
    f(x) = 1 / (1 + exp(-x))
    """
    def __init__(self):
        super().__init__()
    def _formula(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    def _gradient(self, x):
        return self._formula(x) * (1 - self._formula(x))

class Softmax(Activation):
    """
    Softmax activation function
    f(x) = exp(x) / sum(exp(x))
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = self._formula(x)
        if self.training:
            self.cache = x
        return x
    def _formula(self, x: np.ndarray[tuple[Any, ...], np.dtype[Any]]):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        output = e_x / np.sum(e_x,axis=-1, keepdims=True)
        return output
    def _gradient(self, x: np.ndarray[tuple[Any, ...], np.dtype[Any]]):
        """x is the gradient from the next layer
        retruns the gradients of the current layer
        A VECTORIZED IMPLEMENATION FOR SOFTMAX GRADIENTS
        """
        M, C = x.shape # number of examples, number of classes
        diag_matrix = (x.reshape(M, C, 1) * np.eye(C))
        outer_mult = (x.reshape(M, C, 1) @ x.reshape(M, 1, C))
        J = diag_matrix - outer_mult
        return J
    def backward(self, grad_output):
        J = self._gradient(self.cache)
        M, C, _ = J.shape
        return (J @ grad_output.reshape(M, C, 1)).squeeze(-1)


class Tanh(Activation):
    """Hyperbolic Tangent activation function
    f(x) = tanh(x)
    """
    def __init__(self):
        super().__init__()
    def _formula(self, x):
        return np.tanh(x)
    def _gradient(self, x):
        return 1 - np.tanh(x) ** 2

class Linear(Activation):
    """Linear activation function
    f(x) = x
    """
    def __init__(self):
        super().__init__()
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
    print("ReLU Forward:", relu(x))
    print("ReLU Torch Forward:", TorchReLU()(torch.from_numpy(x)))
    # relu.training = True
    relu.backward(np.ones_like(x))
    print("ReLU Backward:", relu.backward(np.ones_like(x)))
    
    sigmoid = Sigmoid()
    print("\nSigmoid Forward:", sigmoid(x))
    print("Sigmoid Torch Forward:", TorchSigmoid()(torch.from_numpy(x)))
    # sigmoid.training = True
    sigmoid.backward(np.ones_like(x))
    print("Sigmoid Backward:", sigmoid.backward(np.ones_like(x)))
    
    tanh = Tanh()
    print("\nTanh Forward:", tanh(x))
    print("Tanh Torch Forward:", TorchTanh()(torch.from_numpy(x)))
    # tanh.training = True
    tanh.backward(np.ones_like(x))
    print("Tanh Backward:", tanh.backward(np.ones_like(x)))
    
    linear = Linear()
    print("\nLinear Forward:", linear(x))
    print("Linear Torch Forward:", TorchLinear()(torch.from_numpy(x)))
    #linear.training = True
    linear.backward(np.ones_like(x))
    print("Linear Backward:", linear.backward(np.ones_like(x)))

def test_softmax():
    import numpy as np
    import time

    # --- Your Implementation (Generalized Jacobian) ---
    def user_gradient(p):
        M, C = p.shape
        # Computes diag(p) - p*p^T for each sample
        diag_matrix = (p.reshape(M, C, 1) * np.eye(C))
        outer_mult = (p.reshape(M, C, 1) @ p.reshape(M, 1, C))
        J = diag_matrix - outer_mult
        return J

    def user_backward(cache, grad_output):
        # cache here is the probability matrix P
        J = user_gradient(cache)
        M, C, _ = J.shape
        return (J @ grad_output.reshape(M, C, 1)).squeeze(-1)

    # --- Optimized Implementation (Specialized Shortcut) ---
    def specialized_backward(P, Y):
        M = P.shape[0]
        return (P - Y) / M

    # --- Helper: Numerical Stability Softmax ---
    def softmax(X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # --- Comparison Setup ---
    M, C = 10, 4
    np.random.seed(42)
    X = np.random.randn(M, C)
    Y = np.eye(C)[np.random.randint(0, C, M)] # Random One-Hot Targets

    # 1. Forward Pass
    s = Softmax()
    P = s.forward(X)

    # 2. Compute "grad_output" for your generalized function
    # This is dL/dP = -Y/P. We divide by M to match the mean loss.
    grad_L_wrt_P = -(Y / (P + 1e-12)) / M

    # 3. Execute both
    grad_user = user_backward(P, grad_L_wrt_P)
    grad_user = s.backward(grad_L_wrt_P)
    grad_specialized = specialized_backward(P, Y)

    # --- Verification ---
    is_equal = np.allclose(grad_user, grad_specialized)
    print(f"Mathematical Equivalence Match: {is_equal}")

    if is_equal:
        print(f"Difference (Max Absolute Error): {np.max(np.abs(grad_user - grad_specialized))}")

    # --- Performance Benchmarking (Scaling to C=1000) ---
    M_large, C_large = 10, 3
    P_large = softmax(np.random.randn(M_large, C_large))
    Y_large = np.eye(C_large)[np.random.randint(0, C_large, M_large)]
    grad_L_wrt_P_large = -(Y_large / (P_large + 1e-12)) / M_large

    start = time.time()
    _ = user_backward(P_large, grad_L_wrt_P_large)
    time_user = time.time() - start

    start = time.time()
    _ = specialized_backward(P_large, Y_large)
    time_specialized = time.time() - start

    print(f"\n--- Performance (M={M_large}, C={C_large}) ---")
    print(f"Generalized Approach: {time_user:.5f}s")
    print(f"Specialized Approach: {time_specialized:.5f}s")
    print(f"Speedup: {time_user / time_specialized:.1f}x")
    
if __name__ == "__main__":
    testing()
    test_softmax()