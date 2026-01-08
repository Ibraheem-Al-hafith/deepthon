
"""
Performance Metrics for Model Evaluation.

This module provides metrics for both Classification (Accuracy, Precision, etc.)
and Regression (MSE, RMSE, R-Squared). It includes a robust preprocessing 
base to handle logits, probabilities, and continuous values.

Classes:
    BaseMetric: Abstract base handling data preparation.
    Classification:
        Accuracy: Calculates the ratio of correct predictions.
        Precision: Measures the quality of positive predictions per class.
        Recall: Measures the ability to find all positive instances per class.
        FBetaScore: Computes the harmonic mean of precision and recall.
    Regression:
        MSE: Mean Squared Error for regression.
        RMSE: Root Mean Squared Error for regression.
        RSquared: Coefficient of Determination for regression.
"""

from typing import Any, Tuple, List, Union, Literal
import numpy as np
from ..nn.activations import Sigmoid, Softmax

# Type alias for floating point NumPy arrays
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]

# =============================================================================
# BASE METRIC INTERFACE
# =============================================================================

class BaseMetric:
    """
    Base class for all performance metrics.
    """

    def __init__(self, task: Literal["classification", "regression"] = "classification") -> None:
        self.task = task

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> Union[float, np.floating]:
        prepared_preds, prepared_true = self._prepare(y_true, y_pred)
        return self._formula(prepared_true, prepared_preds)

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> Any:
        raise NotImplementedError("Subclasses must implement the _formula method.")

    def _prepare(self, y_true: NDArray, y_pred: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Normalizes data. For regression, it ensures shapes match.
        For classification, it handles logits/one-hot conversion.
        """
        if self.task == "regression":
            return y_pred, y_true

        # Classification Logic (Thresholding/Argmax)
        is_binary: bool = (y_true.shape[1] == 1)
        if is_binary:
            is_prob: np.bool = np.all((y_pred >= 0.0) & (y_pred <= 1.0))
            probs: NDArray = y_pred if is_prob else Sigmoid().forward(y_pred)
            preds: NDArray = (probs > 0.5).astype(np.float32)
        else:
            is_prob = np.all((y_pred >= 0.0) & (y_pred <= 1.0))
            probs = y_pred if is_prob else Softmax().forward(y_pred)
            preds = np.argmax(probs, axis=-1, keepdims=True)

        true_labels = np.argmax(y_true, axis=-1, keepdims=True) if y_true.shape[1] > 1 else y_true
        return preds, true_labels

# =============================================================================
# METRIC IMPLEMENTATIONS
# =============================================================================

class Accuracy(BaseMetric):
    """
    Calculates the Accuracy score.
    Formula: (TP + TN) / (TP + TN + FP + FN)
    """
    def __init__(self):
        super().__init__(task="classification")

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        """Computes the mean of correct matches."""
        return np.mean(y_true == y_pred)


class Precision(BaseMetric):
    """
    Calculates the Macro-Averaged Precision score.
    
    Precision is the ability of the classifier not to label as positive 
    a sample that is negative.
    
    """
    def __init__(self):
        super().__init__(task="classification")

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        classes: NDArray = np.unique(y_true)
        class_precisions: List[float] = []
        
        for c in classes:
            tp: int = np.sum((y_true == c) & (y_pred == c))
            fp: int = np.sum((y_true != c) & (y_pred == c))
            
            score: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_precisions.append(score)
            
        return np.mean(class_precisions)


class Recall(BaseMetric):
    """
    Calculates the Macro-Averaged Recall score.
    
    Recall is the ability of the classifier to find all the positive samples.
    """
    def __init__(self):
        super().__init__(task="classification")

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        classes: NDArray = np.unique(y_true)
        class_recalls: List[float] = []
        
        for c in classes:
            tp: int = np.sum((y_true == c) & (y_pred == c))
            fn: int = np.sum((y_true == c) & (y_pred != c))
            
            score: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_recalls.append(score)
            
        return np.mean(class_recalls)


class FBetaScore(BaseMetric):
    """
    Calculates the F-Beta score.
    
    The F-beta score is the weighted harmonic mean of precision and recall.
    F1-score is the special case where beta = 1.0.

    Attributes:
        beta_sq (float): The squared beta parameter.
    """

    def __init__(self, beta: float = 1.0) -> None:
        """
        Args:
            beta (float): Weight of recall in the harmonic mean. Defaults to 1.0.
        """
        super().__init__(task="classification")
        self.beta_sq: float = beta ** 2
        self._precision_metric = Precision()
        self._recall_metric = Recall()

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> Union[float, np.floating]:
        """Computes the weighted harmonic mean of precision and recall."""
        # Note: We pass through the internal _formula to avoid double-calling _prepare
        p: np.floating = self._precision_metric._formula(y_true, y_pred)
        r: np.floating = self._recall_metric._formula(y_true, y_pred)
        
        if (p + r) == 0:
            return 0.0
            
        score: np.floating = (1 + self.beta_sq) * (p * r) / ((self.beta_sq * p) + r)
        return score
    

# =============================================================================
# REGRESSION METRICS
# =============================================================================

class MAE(BaseMetric):
    """
    Mean Absolute Error (MAE).
    Formula: (1/n) * Σ|y_true - y_pred|
    """
    def __init__(self) -> None:
        super().__init__(task="regression")

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        return np.mean(np.abs(y_true - y_pred))
    
class MSE(BaseMetric):
    """
    Mean Squared Error (MSE).
    Formula: (1/n) * Σ(y_true - y_pred)^2
    """
    def __init__(self) -> None:
        super().__init__(task="regression")

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        return np.mean(np.square(y_true - y_pred))


class RMSE(MSE):
    """
    Root Mean Squared Error (RMSE).
    Formula: sqrt(MSE)
    """
    def __init__(self) -> None:
        super().__init__()
    def _formula(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        mse = super()._formula(y_true, y_pred)
        return np.sqrt(mse)


class RSquared(BaseMetric):
    """
    R-Squared (Coefficient of Determination).
    Measures the proportion of variance explained by the model.
    """
    def __init__(self) -> None:
        super().__init__(task="regression")

    def _formula(self, y_true: NDArray, y_pred: NDArray) -> Union[float, np.floating]:
        ss_res = np.sum(np.square(y_true - y_pred))
        ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
            
        return 1 - (ss_res / ss_tot)