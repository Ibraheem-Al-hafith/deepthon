"""
Basic layer components for neural network construction.

This module provides the structural building blocks for multi-layer perceptrons, 
including weight initialization strategies, dense layers, regularization 
techniques (Dropout), and normalization (Batch Norm).

Classes:
    Layer: Dense fully-connected layer with integrated activation.
    Dropout: Regularization layer for preventing overfitting.
    BatchNorm: Batch normalization for accelerating deep network training.
    Sequential: Container for stacking modules in a linear pipeline.

Functions:
    HE(shape): Kaiming initialization for ReLU-based networks.
    Xavier(shape, dist): Glorot initialization for Sigmoid/Tanh-based networks.
"""

from typing import Literal, Tuple, List, Any, Dict, Optional, Union, Callable
import numpy as np
from .activations import ReLU, Tanh, Sigmoid, Activation, Linear
from .base import Module

# Type alias for float-based NumPy arrays of any shape
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]

# =====================================================================
# UTILITIES: INITIALIZATION & FACTORIES
# =====================================================================

def HE(shape: Tuple[int, int]) -> NDArray:
    """
    HE initialization (Kaiming Init).
    
    Optimized for ReLU activations to maintain variance across layers and 
    prevent vanishing or exploding gradients.

    Args:
        shape (Tuple[int, int]): Shape of the weights matrix (n_in, n_out).

    Returns:
        NDArray: Weights sampled from a normal distribution scaled by sqrt(2/n_in).
    """
    n_in: int = shape[0]
    scale: float = np.sqrt(2 / n_in)
    weights: NDArray = np.random.randn(*shape) * scale
    return weights

def Xavier(shape: Tuple[int, int], dist: Literal["normal", "uniform"] = "normal") -> NDArray:
    """
    Xavier initialization (Glorot Init).
    
    Optimized for Sigmoid and Tanh activations by keeping the variance 
    of the input and output gradients similar.

    Args:
        shape (Tuple[int, int]): Shape of the weights matrix (n_in, n_out).
        dist (Literal["normal", "uniform"]): Distribution type to sample from.

    Returns:
        NDArray: Initialized weights matrix.

    Raises:
        ValueError: If 'dist' is not 'normal' or 'uniform'.
    """
    n_in: int = shape[0]
    n_out: int = shape[1]
    
    if dist == "normal":
        scale: float = np.sqrt(2 / (n_in + n_out))
        return np.random.randn(*shape) * scale
    elif dist == "uniform":
        limit: float = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(low=-limit, high=limit, size=shape)
    else:
        raise ValueError("The value of dist should be either 'normal' or 'uniform'")

# Factory mapping for weight initialization strategies
initialize: Dict[str, Callable[[Tuple[int, int]], NDArray]] = {
    "random": lambda shape: np.random.randn(*shape) * 0.01,
    "relu": HE,
    "sigmoid": Xavier,
    "tanh": Xavier,
}

# Factory mapping for activation instances
get_activation: Dict[str, Callable[[], Activation]] = {
    "relu": lambda: ReLU(),
    "sigmoid": lambda: Sigmoid(),
    "tanh": lambda: Tanh(),
    "linear": lambda: Linear(),
}

# =====================================================================
# CORE LAYER
# =====================================================================

class Layer(Module):
    """
    Dense (Fully Connected) Layer.
    
    Performs a linear transformation followed by a non-linear activation.
    Formula: a = activation(x @ W + b)

    Attributes:
        weights (NDArray): The learnable weight matrix.
        bias (NDArray): The learnable bias vector.
        activation (Activation): The activation function module.
        x (Optional[NDArray]): Cached input for backpropagation.
        dw (Optional[NDArray]): Gradient of the loss with respect to weights.
        db (Optional[NDArray]): Gradient of the loss with respect to biases.

    Methods:
        forward(x): Computes the linear and non-linear transformation.
        backward(grad_output): Computes gradients for x, weights, and biases.
        get_parameters(): Returns a list of parameter and gradient dictionaries.
    """

    def __init__(
        self, 
        n_inputs: int, 
        n_neurons: int,
        activation: Union[Literal["relu", "sigmoid", "tanh", "linear"], Activation, None] = None,
    ) -> None:
        """
        Args:
            n_inputs (int): Number of input features.
            n_neurons (int): Number of neurons in this layer.
            activation (Union[str, Activation, None]): Activation function to apply.
        """
        super().__init__()
        
        # 1. Determine initialization strategy based on activation
        init_key: str = str(activation) if isinstance(activation, str) and activation in initialize else \
                        "sigmoid" if n_neurons == 1 else "relu"
        
        self.weights: NDArray = initialize[init_key]((n_inputs, n_neurons))
        self.bias: NDArray = np.zeros((1, n_neurons))
        
        # 2. Setup Activation Module
        if isinstance(activation, Activation):
            self.activation: Activation = activation
        else:
            self.activation = get_activation[activation or "linear"]()
            
        # 3. State storage for backpropagation
        self.x: Optional[NDArray] = None
        self.dw: Optional[NDArray] = None
        self.db: Optional[NDArray] = None

    def forward(self, x: NDArray) -> NDArray:
        """
        Performs the forward pass through the dense layer.
        
        Args:
            x (NDArray): Input data of shape (Batch Size, Input Features).
            
        Returns:
            NDArray: Layer activations of shape (Batch Size, Neurons).
        """
        # Linear transform: z = xW + b
        z: NDArray = x @ self.weights + self.bias
        a: NDArray = self.activation(z)
        
        if self.training:
            self.x = x
            
        return a

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        Backward pass using the chain rule to compute local and upstream gradients.
        
        Args:
            grad_output (NDArray): Upstream gradient (dL/da).
            
        Returns:
            NDArray: Gradient with respect to the input x (dL/dx).

        Raises:
            RuntimeError: If backward is called without a prior forward pass in training mode.
        """
        if self.x is None:
            raise RuntimeError("Backward called before forward or while not in training mode.")
            
        # Compute gradient through the activation function (dL/dz)
        dL_dz: NDArray = self.activation.backward(grad_output)
        
        # Gradients for parameters: dw = x.T @ dz, db = sum(dz)
        self.dw = self.x.T @ dL_dz
        self.db = np.sum(dL_dz, axis=0, keepdims=True)
        
        # Gradient to pass to the previous layer: dx = dz @ W.T
        dL_dx: NDArray = dL_dz @ self.weights.T
        return dL_dx

    def get_parameters(self) -> List[Dict[str, Any]]:
        """
        Returns the layer's weights and biases with their respective gradients.

        Returns:
            List[Dict[str, Any]]: List containing parameter, gradient, and name.
        """
        return [
            {"param": self.weights, "grad": self.dw, "name": "weights"},
            {"param": self.bias, "grad": self.db, "name": "bias"}
        ]

# =====================================================================
# REGULARIZATION: DROP OUT
# =====================================================================

class Dropout(Module):
    """
    Dropout Layer for regularization.
    
    Randomly zeros out input elements with probability p during training. 
    Uses "Inverted Dropout" where active neurons are scaled by 1/keep_prob.

    Attributes:
        drop_prob (float): Probability of zeroing an element.
        keep_prob (float): Probability of keeping an element.
        mask (Optional[NDArray]): Binary mask applied during the forward pass.

    Methods:
        forward(x): Applies the drop mask and scales inputs.
        backward(grad): Propagates the gradient through the mask.
    """

    def __init__(self, p: float = 0.2) -> None:
        """
        Args:
            p (float): Probability of dropping a neuron (0 to 1).
        """
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError("Drop probability must be between 0 and 1")
        self.drop_prob: float = p
        self.keep_prob: float = 1 - p
        self.mask: Optional[NDArray] = None

    def forward(self, x: NDArray) -> NDArray:
        """
        Inverted Dropout forward pass. Scales values by 1/keep_prob during training.

        Args:
            x (NDArray): Input tensor.

        Returns:
            NDArray: Masked and scaled tensor if training, otherwise original tensor.
        """
        if not self.training:
            return x
        
        # Create a random mask and scale it immediately
        self.mask = (np.random.rand(*x.shape) < self.keep_prob).astype(np.float32)
        return (x * self.mask) / self.keep_prob

    def backward(self, grad: NDArray) -> NDArray:
        """
        Dropout backward pass.

        Args:
            grad (NDArray): Upstream gradient.

        Returns:
            NDArray: Gradient zeroed out where neurons were dropped.
        """
        if self.training and self.mask is not None:
            return grad * self.mask / self.keep_prob
        return grad

# =====================================================================
# NORMALIZATION: BATCH NORM
# =====================================================================

class BatchNorm(Module):
    """
    Batch Normalization Layer.
    
    Normalizes inputs to have zero mean and unit variance per feature,
    followed by a learnable scaling (gamma) and shifting (beta).

    Attributes:
        gamma (NDArray): Learnable scaling parameter.
        beta (NDArray): Learnable shifting parameter.
        momentum (float): Momentum factor for running mean/variance computation.
        epsilon (float): Small constant for numerical stability.
        running_mean (NDArray): Exponential moving average of feature means.
        running_var (NDArray): Exponential moving average of feature variances.
        dgamma (Optional[NDArray]): Gradient with respect to gamma.
        dbeta (Optional[NDArray]): Gradient with respect to beta.
        X_hat (Optional[NDArray]): Normalized input stored for backward pass.
        std_inv (Optional[NDArray]): Inverse standard deviation stored for backward pass.

    Methods:
        forward(X): Normalizes batch and updates running statistics.
        backward(grad): Computes gradients for gamma, beta, and input X.
        get_parameters(): Returns learnable parameters and their gradients.
    """

    def __init__(self, num_features: int, momentum: float = 0.9, epsilon: float = 1e-5) -> None:
        """
        Args:
            num_features (int): Number of features (neurons) in the input.
            momentum (float): Momentum for exponential moving averages.
            epsilon (float): Epsilon for numerical stability in sqrt.
        """
        super().__init__()
        self.gamma: NDArray = np.ones((1, num_features))
        self.beta: NDArray = np.zeros((1, num_features))
        self.momentum: float = momentum
        self.epsilon: float = epsilon
        
        self.running_mean: NDArray = np.zeros((1, num_features))
        self.running_var: NDArray = np.ones((1, num_features))
        
        self.dgamma: Optional[NDArray] = None
        self.dbeta: Optional[NDArray] = None
        
        self.X_hat: Optional[NDArray] = None
        self.std_inv: Optional[NDArray] = None

    def forward(self, X: NDArray) -> NDArray:
        """
        Forward pass for BatchNorm.

        Args:
            X (NDArray): Input batch of shape (Batch Size, Features).

        Returns:
            NDArray: Normalized and scaled batch.
        """
        if self.training:
            # Calculate batch statistics
            batch_mean: NDArray = np.mean(X, axis=0, keepdims=True)
            batch_var: NDArray = np.maximum(np.var(X, axis=0, keepdims=True), 0.0)
            
            # Update global running stats for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            X_centered: NDArray = X - batch_mean
            self.std_inv = 1.0 / np.sqrt(batch_var + self.epsilon)
            self.X_hat = X_centered * self.std_inv
        else:
            # Use running stats for inference
            X_centered = X - self.running_mean
            std_inv = 1.0 / np.sqrt(self.running_var + self.epsilon)
            self.X_hat = X_centered * std_inv

        return self.gamma * self.X_hat + self.beta

    def backward(self, grad: NDArray) -> NDArray:
        """
        Backward pass for BatchNorm using the simplified chain rule.

        Args:
            grad (NDArray): Upstream gradient.

        Returns:
            NDArray: Gradient with respect to the input X.
        """
        if self.X_hat is None or self.std_inv is None:
            raise RuntimeError("Backward called before forward.")
            
        batch_size: int = grad.shape[0]
        
        # Gradients for learnable parameters
        self.dgamma = np.sum(grad * self.X_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(grad, axis=0, keepdims=True)
        
        # Gradient for the input data (da/dX)
        dx_hat: NDArray = grad * self.gamma
        da: NDArray = (1.0 / batch_size) * self.std_inv * (
            batch_size * dx_hat - np.sum(dx_hat, axis=0, keepdims=True) -
            self.X_hat * np.sum(dx_hat * self.X_hat, axis=0, keepdims=True)
        )
        return da

    def get_parameters(self) -> List[Dict[str, Any]]:
        """Returns gamma and beta parameters."""
        return [
            {"param": self.gamma, "grad": self.dgamma, "name": "gamma"},
            {"param": self.beta, "grad": self.dbeta, "name": "beta"}
        ]

# =====================================================================
# CONTAINER: SEQUENTIAL
# =====================================================================

class Sequential(Module):
    """
    Sequential Container for linear stacks of layers.
    
    Manages data flow and propagates state (training/evaluation) recursively 
    through all child modules.

    Attributes:
        layers (List[Module]): Ordered list of layers in the model.

    Methods:
        train(): Sets all child layers to training mode.
        eval(): Sets all child layers to evaluation mode.
        forward(x): Propagates input through all layers in order.
        backward(grad): Propagates gradient through all layers in reverse order.
        add(layers): Appends one or more modules to the stack.
    """

    def __init__(self, layers: Optional[List[Module]] = None) -> None:
        """
        Args:
            layers (Optional[List[Module]]): Initial list of modules.
        """
        super().__init__()
        self.layers: List[Module] = layers if layers is not None else []

    def train(self) -> None:
        """Sets the container and all constituent layers to training mode."""
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        """Sets the container and all constituent layers to evaluation mode."""
        self.training = False
        for layer in self.layers:
            layer.eval()

    def forward(self, x: NDArray) -> NDArray:
        """
        Forward pass through the sequential stack.

        Args:
            x (NDArray): Input tensor.

        Returns:
            NDArray: Output of the final layer in the sequence.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: NDArray) -> NDArray:
        """
        Backward pass through the stack in reverse order.

        Args:
            grad (NDArray): Upstream gradient from the loss function.

        Returns:
            NDArray: Gradient with respect to the initial input.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def add(self, layers: Union[Module, List[Module]]) -> None:
        """
        Appends new layer(s) to the sequential stack.

        Args:
            layers (Union[Module, List[Module]]): A single module or list of modules to append.
        """
        if isinstance(layers, list):
            for layer in layers:
                self.add(layer)
        else:
            self.layers.append(layers)