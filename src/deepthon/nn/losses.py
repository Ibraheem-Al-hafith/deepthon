"""
Loss functions for neural network optimization.

This module provides a suite of common objective functions used to measure 
model performance during training. It includes implementations for regression 
(MSE) and classification (Binary and Categorical Cross-Entropy).

Classes:
    LOSS: Abstract base class for all loss functions.
    MSE: Mean Squared Error for regression tasks.
    MAE: Mean Absolute Error for regression tasks.
    BCE: Binary Cross-Entropy for binary classification.
    CrossEntropy: Categorical Cross-Entropy for multi-class classification.
"""

import numpy as np
from typing import Any, Union

# Type alias for floating point NumPy arrays of any shape
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]


# =============================================================================
# BASE LOSS INTERFACE
# =============================================================================

class LOSS:
    """
    Base class for all loss functions.
    
    A loss function measures the discrepancy between the predicted output 
    and the ground truth. This class provides the common interface for 
    forward (loss calculation) and backward (gradient calculation) passes.

    Attributes:
        None

    Methods:
        __call__(y_true, y_pred): Syntactic sugar for the forward pass.
        forward(y_true, y_pred): Abstract method for scalar loss calculation.
        backward(y_true, y_pred): Abstract method for gradient calculation.
    """

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> Union[float, np.floating]:
        """
        Allows the loss instance to be called like a function.

        Args:
            y_true (NDArray): Ground truth labels.
            y_pred (NDArray): Predicted values from the model.

        Returns:
            Union[float, np.floating]: The calculated scalar loss value.
        """
        return self.forward(y_true, y_pred)

    def forward(self, y_true: NDArray, y_pred: NDArray) -> Union[float, np.floating]:
        """
        Calculate the scalar loss value.

        Args:
            y_true (NDArray): Ground truth labels.
            y_pred (NDArray): Predicted values (probabilities or logits).

        Returns:
            Union[float, np.floating]: Scalar loss value.

        Raises:
            NotImplementedError: If the subclass does not implement the method.
        """
        raise NotImplementedError("Loss functions must implement the forward method.")

    def backward(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """
        Calculate the gradient of the loss with respect to the predictions.

        Args:
            y_true (NDArray): Ground truth labels.
            y_pred (NDArray): Predicted values.

        Returns:
            NDArray: Gradient tensor (dL/dy_pred).

        Raises:
            NotImplementedError: If the subclass does not implement the method.
        """
        raise NotImplementedError("Loss functions must implement the backward method.")


# =============================================================================
# REGRESSION LOSSES
# =============================================================================

class MSE(LOSS):
    """
    Mean Squared Error (MSE) Loss.
    
    Commonly used for regression tasks where the goal is to predict 
    continuous values.
    Formula: L = (1/n) * Σ(y_true - y_pred)^2

    Attributes:
        None

    Methods:
        forward(y_true, y_pred): Computes the mean squared difference.
        backward(y_true, y_pred): Computes the gradient (2/m) * (y_pred - y_true).
    """

    def forward(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        """
        Calculate the MSE loss.

        Args:
            y_true (NDArray): Target values.
            y_pred (NDArray): Predicted values.

        Returns:
            np.floating: The mean of the squared errors.
        """
        # Calculate element-wise square difference and then the mean
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """
        Calculate the gradient of MSE loss.

        The derivative of (1/m) * (y_t - y_p)^2 w.r.t y_p is (2/m) * (y_p - y_t).

        Args:
            y_true (NDArray): Target values.
            y_pred (NDArray): Predicted values.

        Returns:
            NDArray: Gradient of the loss with respect to y_pred.
        """
        m: int = y_true.size
        return (2 / m) * (y_pred - y_true)
class MAE(LOSS):
    """
    Mean Absolute Error (MAE) Loss.
    
    Commonly used for regression tasks where the goal is to predict 
    continuous values.
    Formula: L = (1/n) * Σ|y_true - y_pred|

    Attributes:
        None

    Methods:
        forward(y_true, y_pred): Computes the mean squared difference.
        backward(y_true, y_pred): Computes the gradient (2/m) * (y_pred - y_true).
    """

    def forward(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        """
        Calculate the MSE loss.

        Args:
            y_true (NDArray): Target values.
            y_pred (NDArray): Predicted values.

        Returns:
            np.floating: The mean of the squared errors.
        """
        # Calculate element-wise square difference and then the mean
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """
        Calculate the gradient of MSE loss.

        The derivative of (1/m) * (y_t - y_p)^2 w.r.t y_p is (2/m) * (y_p - y_t).

        Args:
            y_true (NDArray): Target values.
            y_pred (NDArray): Predicted values.

        Returns:
            NDArray: Gradient of the loss with respect to y_pred.
        """
        m: int = y_true.size
        return (1 / m) * np.sign(y_pred - y_true)


# =============================================================================
# CLASSIFICATION LOSSES
# =============================================================================

class BCE(LOSS):
    """
    Binary Cross-Entropy (BCE) Loss.
    
    Used for binary classification. Supports stable calculation directly from 
    unnormalized logits to prevent numerical underflow/overflow.

    Attributes:
        from_logits (bool): Indicates if y_pred contains raw scores (logits) 
            or probabilities.

    Methods:
        forward(y_true, y_pred): Computes binary cross-entropy.
        backward(y_true, y_pred): Computes the gradient of the loss.
    """

    def __init__(self, from_logits: bool = True) -> None:
        """
        Args:
            from_logits (bool): If True, y_pred is assumed to be raw scores. 
                Defaults to True.
        """
        self.from_logits: bool = from_logits

    def forward(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        """
        Calculate the BCE loss.

        Args:
            y_true (NDArray): Binary ground truth (0 or 1).
            y_pred (NDArray): Predicted values.

        Returns:
            np.floating: Mean binary cross-entropy loss.
        """
        if self.from_logits:
            # Stable BCE implementation: max(x, 0) - x * y + log(1 + exp(-abs(x)))
            # This prevents overflow in the exp() function.
            loss: NDArray = np.maximum(y_pred, 0) - y_pred * y_true + \
                            np.log(1 + np.exp(-np.abs(y_pred)))
            return np.mean(loss)

        # Probability-based implementation with epsilon clipping for log stability
        epsilon: float = 1e-7
        y_pred_clipped: NDArray = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    def backward(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """
        Calculate the gradient of BCE loss.

        Args:
            y_true (NDArray): Binary ground truth labels.
            y_pred (NDArray): Predicted values.

        Returns:
            NDArray: Gradient w.r.t y_pred.
        """
        m: int = y_true.shape[0]

        if self.from_logits:
            # Shortcut for d(BCE(Sigmoid(logits))) / d(logits): (sigmoid(x) - y) / m
            sig: NDArray = 1 / (1 + np.exp(-np.clip(y_pred, -500, 500)))
            return (sig - y_true) / m

        # Gradient with respect to probabilities: (1/m) * (p - y) / (p * (1 - p))
        epsilon: float = 1e-7
        y_pred_clipped: NDArray = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return (1 / m) * ((y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped)))


class CrossEntropy(LOSS):
    """
    Categorical Cross-Entropy Loss.
    
    Used for multi-class classification tasks. Features the 'Log-Sum-Exp' 
    numerical stability trick for handling unnormalized logits.

    Attributes:
        from_logits (bool): Indicates if y_pred contains raw scores (logits).

    Methods:
        forward(y_true, y_pred): Computes multi-class cross-entropy.
        backward(y_true, y_pred): Computes the gradient w.r.t y_pred.
    """

    def __init__(self, from_logits: bool = True) -> None:
        """
        Args:
            from_logits (bool): If True, treats y_pred as unnormalized logits.
                Defaults to True.
        """
        self.from_logits: bool = from_logits

    def forward(self, y_true: NDArray, y_pred: NDArray) -> np.float64:
        """
        Calculate the Categorical Cross-Entropy loss.

        Args:
            y_true (NDArray): One-hot encoded ground truth.
            y_pred (NDArray): Predicted values (logits or probabilities).

        Returns:
            np.float64: Mean categorical cross-entropy.
        """
        epsilon: float = 1e-15

        if self.from_logits:
            # Convert logits to probabilities using stable Softmax (max subtraction)
            shift_y_pred: NDArray = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exps: NDArray = np.exp(shift_y_pred)
            softmax_probs: NDArray = exps / np.sum(exps, axis=1, keepdims=True)
            y_pred = softmax_probs

        # Clip values to avoid log(0)
        y_pred_clipped: NDArray = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def backward(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """
        Calculate the gradient of Categorical Cross-Entropy loss.

        Args:
            y_true (NDArray): One-hot encoded labels.
            y_pred (NDArray): Predicted values.

        Returns:
            NDArray: Gradient of the loss w.r.t y_pred.
        """
        m: int = y_true.shape[0]
        epsilon: float = 1e-15

        if self.from_logits:
            # Shortcut for d(CE(Softmax(logits))) / d(logits): (softmax(p) - y) / m
            shift_y_pred: NDArray = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exps: NDArray = np.exp(shift_y_pred)
            softmax_probs: NDArray = exps / np.sum(exps, axis=1, keepdims=True)
            return (softmax_probs - y_true) / m

        # Gradient w.r.t probabilities: - (y_true / y_pred) / m
        y_pred_clipped: NDArray = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return - (y_true / y_pred_clipped) / m