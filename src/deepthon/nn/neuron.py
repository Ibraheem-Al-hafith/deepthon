"""
Single Neuron implementation for fundamental neural network operations.

This module defines a standalone Neuron class capable of performing forward 
propagation, backpropagation, and weight optimization (fitting) using 
gradient descent. It is designed to mirror the behavior of a single 
unit in a dense layer.

Classes:
    Neuron: A single computational unit with weight management and training logic.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from .activations import ReLU, Sigmoid, Tanh, Linear, Activation
from .base import Module
from .losses import BCE, LOSS

# Type alias for float-based NumPy arrays
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]

# Factory mapping for activation instances
get_activation: Dict[str, Callable[[], Activation]] = {
    "relu": lambda: ReLU(),
    "sigmoid": lambda: Sigmoid(),
    "tanh": lambda: Tanh(),
    "linear": lambda: Linear(),
}


class Neuron(Module):
    """
    A single neuron in a neural network.
    
    The neuron performs a weighted sum of inputs plus a bias, followed by 
    a non-linear activation function. It maintains its own state for 
    training via gradient descent.

    Attributes:
        input_size (int): The number of features expected in the input.
        weights (NDArray): Learnable weight matrix of shape (input_size, 1).
        bias (NDArray): Learnable bias vector of shape (1, 1).
        activation (Activation): The activation function module applied to the output.
        cache (Optional[NDArray]): Stores the input data from the last forward pass 
            for gradient calculation.
        history (List[float]): A list of loss values recorded during the fit process.

    Methods:
        forward(x): Computes the activated output of the neuron.
        backward(grad_output): Computes gradients for weights, bias, and input.
        fit(x, y, loss_fn, lr, epochs): Optimizes weights using gradient descent.
        predict(x): Performs inference without caching data.
    """

    def __init__(self, input_size: int, activation: str = "sigmoid") -> None:
        """
        Args:
            input_size (int): Number of input features.
            activation (str): Key for the activation function (default: "sigmoid").
        """
        super().__init__()
        self.input_size: int = input_size
        
        # Initialize weights with small random values to break symmetry
        self.weights: NDArray = np.random.randn(input_size, 1) * 0.01
        self.bias: NDArray = np.zeros((1, 1))
        
        # Set up the non-linear component
        self.activation: Activation = get_activation[activation.lower()]()
        
        # State for training
        self.cache: Optional[NDArray] = None
        self.history: List[float] = []

    def forward(self, x: NDArray) -> NDArray:
        """
        Computes the forward pass: a = activation(x @ W + b).

        Args:
            x (NDArray): Input features of shape (m_samples, n_features).

        Returns:
            NDArray: Activated output of shape (m_samples, 1).
        """
        # Save input for backward pass if in training mode
        if self.training:
            self.cache = x
            
        # 1. Linear combination: z = xW + b
        # Resulting shape: (m, n) @ (n, 1) + (1, 1) -> (m, 1)
        z: NDArray = x @ self.weights + self.bias
        
        # 2. Non-linear transformation
        a: NDArray = self.activation.forward(z)
        return a

    def backward(self, grad_output: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Computes the backward pass using the chain rule.

        Args:
            grad_output (NDArray): Upstream gradient from the loss function (dL/da).

        Returns:
            Tuple[NDArray, NDArray, NDArray]: Gradients (dL_dw, dL_db, dL_dx).

        Raises:
            RuntimeError: If cache is empty (forward was not called).
        """
        if self.cache is None:
            raise RuntimeError("Backward called before forward pass.")

        # 1. Gradient through activation: dL/dz = (dL/da) * (da/dz)
        da_dz: NDArray = self.activation.backward(grad_output=grad_output)
        dL_dz: NDArray = da_dz 

        # 2. Gradient w.r.t weights: dL/dw = x.T @ dL/dz
        # Shape: (n, m) @ (m, 1) -> (n, 1)
        dL_dw: NDArray = self.cache.T @ dL_dz
        
        # 3. Gradient w.r.t bias: sum of gradients across the batch
        dL_db: NDArray = np.sum(dL_dz, axis=0, keepdims=True)
        
        # 4. Gradient w.r.t input (to pass to earlier layers): dL/dx = dL/dz @ W.T
        # Shape: (m, 1) @ (1, n) -> (m, n)
        dL_dx: NDArray = dL_dz @ self.weights.T

        return dL_dw, dL_db, dL_dx
    
    def fit(
        self, 
        x: NDArray, 
        y: NDArray, 
        loss_fn: LOSS, 
        lr: float = 1e-2, 
        epochs: int = 100
    ) -> List[float]:
        """
        Trains the neuron using standard Gradient Descent.

        Args:
            x (NDArray): Training features.
            y (NDArray): Training targets (ground truth).
            loss_fn (LOSS): Loss function instance (e.g., BCE).
            lr (float): Learning rate for weight updates.
            epochs (int): Number of complete passes over the dataset.

        Returns:
            List[float]: The loss history over the training duration.
        """
        self.history = []
        
        for epoch in range(epochs):
            # 1. Forward pass to get predictions
            y_pred: NDArray = self.forward(x)

            # 2. Calculate current loss
            loss: float = float(loss_fn(y, y_pred))
            self.history.append(loss)
            
            # 3. Backward pass to compute gradients
            # First get dL/dy_pred from loss function, then propagate through neuron
            grad_loss: NDArray = loss_fn.backward(y, y_pred)
            dL_dw, dL_db, _ = self.backward(grad_loss)

            # 4. Update parameters: Param = Param - (LR * Gradient)
            self.weights -= lr * dL_dw
            self.bias -= lr * dL_db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

        return self.history

    def predict(self, x: NDArray) -> NDArray:
        """
        Performs inference on new data. Unlike forward(), this does not cache inputs.

        Args:
            x (NDArray): Input features.

        Returns:
            NDArray: Predicted probabilities or values.
        """
        return self.activation.forward(x @ self.weights + self.bias)


# =============================================================================
# DEMONSTRATION & COMPARISON WITH SCIKIT-LEARN
# =============================================================================

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    # 1. Generate Synthetic Binary Classification Data
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.1,
        class_sep=1.5,
        random_state=42
    )
    
    # Reshape targets to (Samples, 1) for matrix operations
    y_reshaped: NDArray = y.reshape(-1, 1).astype(np.float64)
    X_float: NDArray = X.astype(np.float64)

    # 2. Setup Custom Neuron and Loss
    input_dim: int = X_float.shape[1]
    my_neuron: Neuron = Neuron(input_size=input_dim, activation="sigmoid")
    loss_function: BCE = BCE(from_logits=False)

    # 3. Train Custom Neuron
    print("--- Training Custom Neuron ---")
    history_log: List[float] = my_neuron.fit(X_float, y_reshaped, loss_function, lr=0.1, epochs=2000)

    # 4. Train Scikit-Learn Logistic Regression for benchmarking
    print("\n--- Training Scikit-Learn LogisticRegression ---")
    sk_model: LogisticRegression = LogisticRegression(C=np.inf)
    sk_model.fit(X_float, y)

    # 5. Result Comparison
    print("\n--- Parameter Comparison ---")
    print(f"Custom Weights: {my_neuron.weights.flatten()}")
    print(f"Sklearn Weights: {sk_model.coef_.flatten()}")
    print(f"Custom Bias:    {my_neuron.bias.flatten()}")
    print(f"Sklearn Bias:    {sk_model.intercept_}")

    # 6. Accuracy Evaluation
    custom_preds: NDArray = (my_neuron.predict(X_float) > 0.5).astype(int)
    sk_preds: NDArray = sk_model.predict(X_float).reshape(-1, 1)

    custom_acc: float = float(np.mean(custom_preds == y_reshaped))
    sk_acc: float = float(np.mean(sk_preds == y_reshaped))

    print(f"\nCustom Accuracy: {custom_acc * 100:.2f}%")
    print(f"Sklearn Accuracy: {sk_acc * 100:.2f}%")