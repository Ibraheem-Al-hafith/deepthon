"""Loss functions for our library"""
import numpy as np
from .activations import Sigmoid
from typing import Any
class LOSS:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return self.forward(y_true, y_pred)
    """Base class for the loss functions"""
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate the loss function"""
        raise NotImplementedError
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate the Gradients"""
        raise NotImplementedError

class MSE(LOSS):
    """
    Calculate mean squared error.
    mse(y, y_hat) = sum((y - y_hat)^2/n)
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
        """Calculate the MSE loss"""
        return np.mean((y_true - y_pred) ** 2)
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[Any]:
        """Calculate the gradient of MSE loss"""
        m = y_true.size
        return (2/m) * (y_pred - y_true)


class BCE(LOSS):
    """Binary Cross Entropy Loss
    BCE(y, y_hat) = -1/n * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
    """
    def __init__(self, from_logits: bool = True) -> None:
        self.from_logits = from_logits

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
        if self.from_logits:
            # y_pred are raw logits (x)
            # Stable formula: max(x, 0) - x * y + log(1 + exp(-abs(x)))
            loss = np.maximum(y_pred, 0) - y_pred * y_true + np.log(1 + np.exp(-np.abs(y_pred)))
            return np.mean(loss)
    
        # If already probabilities, use clip logic with a larger epsilon
        epsilon = 1e-7 
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]
        
        if self.from_logits:
            # The gradient of BCE(Sigmoid(x)) is simply (Sigmoid(x) - y_true)
            # We still need the sigmoid values for this calculation
            sig = 1 / (1 + np.exp(-np.clip(y_pred, -500, 500)))
            return (sig - y_true) / m
        
        # Non-logits version (less stable)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return (1 / m) * ((y_pred - y_true) / (y_pred * (1 - y_pred)))

class CrossEntropy(LOSS):
    """
    Categorical Cross Entropy Loss.
    Formula: L = -1/m * sum(y_true * log(y_pred))
    """
    def __init__(self, from_logits: bool = True) -> None:
        self.from_logits = from_logits

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """Calculate the CCE loss"""
        epsilon = 1e-15
        
        if self.from_logits:
            # Step 1: Stable Softmax (Log-Sum-Exp trick)
            shift_y_pred = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exps = np.exp(shift_y_pred)
            softmax_probs = exps / np.sum(exps, axis=1, keepdims=True)
            y_pred = softmax_probs
            
        # Step 2: Clip to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Step 3: Mean of the sum of logs
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the gradient of CCE loss"""
        m = y_true.shape[0]
        epsilon = 1e-15
        
        if self.from_logits:
            # Shortcut: Gradient of (Softmax + CrossEntropy) w.r.t logits
            shift_y_pred = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exps = np.exp(shift_y_pred)
            softmax_probs = exps / np.sum(exps, axis=1, keepdims=True)
            
            return (softmax_probs - y_true) / m
        
        # Standard derivative w.r.t probabilities
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return - (y_true / y_pred) / m


def test() -> None:
    from sklearn.metrics import mean_squared_error, log_loss
    import numpy as np

    # 1. Test Data
    y_true_cat = np.array([[1, 0, 0], [0, 1, 0]]) # One-hot encoded
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])

    # 2. Your Implementation (Simplified for testing)
    def my_softmax(z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=1, keepdims=True)

    y_pred_cat = my_softmax(logits)
    my_cce = CrossEntropy(from_logits=True).forward(y_true_cat, logits)

    # 3. Scikit-Learn Implementation
    sk_cce = log_loss(y_true_cat, y_pred_cat)

    print(f"Your CCE: {my_cce}")
    print(f"Sklearn CCE: {sk_cce}")
    print(f"Match: {np.allclose(my_cce, sk_cce)}")

    y_true_binary = np.array([1, 0, 1, 1, 0], dtype=np.float64).reshape(-1, 1)
    logits_binary = np.array([0, 0, 1, 1, 0], dtype=np.float64).reshape(-1, 1)

    y_pred_binary = my_softmax(logits_binary)
    my_cce_binary = BCE(from_logits=True).forward(y_true_binary, logits_binary)

    print(f"Your Binary CCE: {my_cce_binary}")
    sk_cce_binary = log_loss(y_true_binary, Sigmoid().forward(logits_binary))
    print(f"Sklearn Binary CCE: {sk_cce_binary}")
    print(f"Match Binary: {np.allclose(my_cce_binary, sk_cce_binary)}")

    y_true_cont = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_cont = np.array([2.5, 0.0, 2.0, 8.0])
    my_mse = MSE().forward(y_true_cont, y_pred_cont)
    sk_mse = mean_squared_error(y_true_cont, y_pred_cont)
    print(f"Your MSE: {my_mse}")
    print(f"Sklearn MSE: {sk_mse}")
    print(f"Match MSE: {np.allclose(my_mse, sk_mse)}")
if __name__ == "__main__":
    test()