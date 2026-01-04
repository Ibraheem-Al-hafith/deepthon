"""
Activation functions for neural networks.

This module provides a suite of common activation functions including ReLU, 
Sigmoid, Tanh, Softmax, and Linear, inheriting from a common base class to 
handle caching and backpropagation logic.

Classes:
    Activation: Abstract base class for all activation layers.
    ReLU: Rectified Linear Unit implementation.
    Sigmoid: Sigmoid logistic function implementation.
    Softmax: Normalized exponential implementation.
    Tanh: Hyperbolic tangent implementation.
    Linear: Identity activation implementation.
"""

from typing import Any, Optional, Type, Union
import numpy as np
from .base import Module

# Custom type alias for clarity in mathematical operations
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]

# =============================================================================
# BASE ACTIVATION INTERFACE
# =============================================================================

class Activation(Module):
    """
    Base class for all activation functions.
    
    Handles standard neural network layer functionality including caching 
    inputs for backpropagation and enforcing formula implementations.

    Attributes:
        cache (Optional[NDArray]): Stores the input or output tensor during 
            the forward pass for use in gradient calculation.
        training (bool): Inherited from Module; indicates if the layer is 
            in training mode.

    Methods:
        forward(x): Performs the forward activation pass.
        backward(grad_output): Computes the gradient using the chain rule.
        __call__(x): Syntactic sugar for the forward pass.
    """

    def __init__(self) -> None:
        """
        Initialize the activation module and its cache.
        """
        super().__init__()
        self.cache: Optional[NDArray] = None

    def __call__(self, x: NDArray) -> NDArray:
        """
        Allows the instance to be called like a function.

        Args:
            x (NDArray): Input tensor to be transformed.

        Returns:
            NDArray: The transformed tensor after activation.
        """
        return self.forward(x)

    def _formula(self, x: NDArray) -> NDArray:
        """
        The mathematical definition of the activation function.

        Args:
            x (NDArray): Input tensor.

        Returns:
            NDArray: Result of applying f(x).

        Raises:
            NotImplementedError: If the subclass does not implement the formula.
        """
        raise NotImplementedError("Activation functions must implement the _formula method.")

    def _gradient(self, x: NDArray) -> NDArray:
        """
        The local derivative of the activation function.

        Args:
            x (NDArray): Input tensor (usually the cached input).

        Returns:
            NDArray: Result of the derivative f'(x).

        Raises:
            NotImplementedError: If the subclass does not implement the gradient.
        """
        raise NotImplementedError("Activation functions must implement the _gradient method.")

    def forward(self, x: NDArray) -> NDArray:
        """
        Performs the forward pass and caches input if in training mode.

        Args:
            x (NDArray): Input tensor of arbitrary shape.

        Returns:
            NDArray: Result of the activation function with the same shape as x.
        """
        # Save input for backward pass if we are training
        if self.training:
            self.cache = x
        return self._formula(x)

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        Performs the backward pass using the chain rule.

        The formula applied is: dL/dx = dL/dy * dy/dx

        Args:
            grad_output (NDArray): Gradient of the loss with respect to the output.

        Returns:
            NDArray: Gradient of the loss with respect to the input (dL/dx).

        Raises:
            RuntimeError: If backward is called before a forward pass has cached data.
        """
        if self.cache is None:
            raise RuntimeError("Backward pass attempted without a cached forward pass.")
        
        # Chain Rule: Upstream Gradient * Local Gradient
        return grad_output * self._gradient(self.cache)


# =============================================================================
# ACTIVATION IMPLEMENTATIONS
# =============================================================================

class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    
    Formula: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0 else 0

    Attributes:
        cache (Optional[NDArray]): Inherited; stores input x.

    Methods:
        _formula(x): Implements the ReLU element-wise maximum.
        _gradient(x): Implements the step function derivative.
    """

    def __init__(self) -> None:
        """
        Initialize the ReLU activation module.
        """
        super().__init__()

    def _formula(self, x: NDArray) -> NDArray:
        """
        Applies ReLU: max(0, x).

        Args:
            x (NDArray): Input tensor.

        Returns:
            NDArray: Tensor where negative values are replaced by zero.
        """
        return np.maximum(0, x)

    def _gradient(self, x: NDArray) -> NDArray:
        """
        Computes the ReLU gradient.

        Args:
            x (NDArray): The cached input tensor.

        Returns:
            NDArray: Binary mask where 1 represents x > 0.
        """
        return (x > 0).astype(x.dtype)


class Sigmoid(Activation):
    """
    Sigmoid activation function.

    Formula: f(x) = 1 / (1 + exp(-x))
    Derivative: f'(x) = f(x) * (1 - f(x))

    Attributes:
        cache (Optional[NDArray]): Inherited; stores input x.

    Methods:
        _formula(x): Implements the logistic sigmoid function.
        _gradient(x): Implements the derivative based on sigmoid output.
    """

    def __init__(self) -> None:
        """
        Initialize the Sigmoid activation module.
        """
        super().__init__()

    def _formula(self, x: NDArray) -> NDArray:
        """
        Applies Sigmoid with numerical clipping to prevent overflow.

        Args:
            x (NDArray): Input tensor.

        Returns:
            NDArray: Squeashed values between 0 and 1.
        """
        # Clipping x to avoid exp overflow in extreme values
        x_clipped: NDArray = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _gradient(self, x: NDArray) -> NDArray:
        """
        Computes Sigmoid gradient using the f(x) * (1 - f(x)) identity.

        Args:
            x (NDArray): The cached input tensor.

        Returns:
            NDArray: Local gradient of the sigmoid.
        """
        sig: NDArray = self._formula(x)
        return sig * (1 - sig)


class Softmax(Activation):
    """
    Softmax activation function with numerical stability.
    
    Used typically for multi-class classification to produce a probability 
    distribution across the last dimension.

    Attributes:
        cache (Optional[NDArray]): In Softmax, this stores the OUTPUT (probabilities)
            rather than the input to simplify the backward calculation.

    Methods:
        forward(x): Overridden to cache output probabilities.
        backward(grad_output): Implements vectorized Jacobian-vector product.
        _formula(x): Implements the stable softmax formula.
    """

    def _formula(self, x: NDArray) -> NDArray:
        """
        Computes Softmax using the 'max subtraction' trick for stability.

        Args:
            x (NDArray): Input tensor of shape (..., Features).

        Returns:
            NDArray: Normalized probability distribution.
        """
        # Shift input by max value for numerical stability (prevents large exp results)
        shift_x: NDArray = x - np.max(x, axis=-1, keepdims=True)
        exps: NDArray = np.exp(shift_x)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, x: NDArray) -> NDArray:
        """
        Forward pass for Softmax. Caches the output (P) for efficiency.

        Args:
            x (NDArray): Input logit tensor.

        Returns:
            NDArray: Probability tensor.
        """
        out: NDArray = self._formula(x)
        if self.training:
            self.cache = out
        return out

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        Optimized vectorized backward pass for Softmax.
        
        Formula: P * (dL/dP - sum(dL/dP * P))

        Args:
            grad_output (NDArray): Upstream gradient.

        Returns:
            NDArray: Gradient with respect to the input logits.
        """
        if self.cache is None:
            raise RuntimeError("Backward called before forward.")

        p: NDArray = self.cache 
        # Calculate the weighted sum of gradients
        sum_grad_p: NDArray = np.sum(grad_output * p, axis=-1, keepdims=True)
        return p * (grad_output - sum_grad_p)


class Tanh(Activation):
    """
    Hyperbolic Tangent activation function.

    Formula: f(x) = tanh(x)
    Derivative: f'(x) = 1 - f(x)^2

    Attributes:
        cache (Optional[NDArray]): Inherited; stores input x.

    Methods:
        _formula(x): Standard hyperbolic tangent.
        _gradient(x): Derivative using the 1 - tanh^2 identity.
    """

    def __init__(self) -> None:
        """
        Initialize the Tanh activation module.
        """
        super().__init__()

    def _formula(self, x: NDArray) -> NDArray:
        """
        Applies np.tanh to the input.

        Args:
            x (NDArray): Input tensor.

        Returns:
            NDArray: Values squashed between -1 and 1.
        """
        return np.tanh(x)

    def _gradient(self, x: NDArray) -> NDArray:
        """
        Computes Tanh gradient: 1 - tanh(x)^2.

        Args:
            x (NDArray): The cached input tensor.

        Returns:
            NDArray: Local gradient of tanh.
        """
        return 1 - np.tanh(x) ** 2


class Linear(Activation):
    """
    Linear (Identity) activation function.

    Formula: f(x) = x
    Derivative: f'(x) = 1

    Attributes:
        cache (Optional[NDArray]): Inherited; stores input x.

    Methods:
        _formula(x): Returns input unchanged.
        _gradient(x): Returns ones.
    """

    def __init__(self) -> None:
        """
        Initialize the Linear activation module.
        """
        super().__init__()

    def _formula(self, x: NDArray) -> NDArray:
        """
        Identity function: returns the input as is.

        Args:
            x (NDArray): Input tensor.

        Returns:
            NDArray: The same input tensor.
        """
        return x

    def _gradient(self, x: NDArray) -> NDArray:
        """
        Returns a gradient of ones with the same shape as input.

        Args:
            x (NDArray): The cached input tensor.

        Returns:
            NDArray: Tensor of ones with shape matching x.
        """
        return np.ones_like(x)