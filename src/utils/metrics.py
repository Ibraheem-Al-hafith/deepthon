import numpy as np
from typing import Any
from ..nnlib.activations import Sigmoid, Softmax

class softmax(Softmax):
    def __init__(self):
        super().__init__()
        self.traininng = False
    def __call__(self, x: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return super().__call__(x)

class sigmoid(Sigmoid):
    def __init__(self):
        super().__init__()
        self.traininng = False
    def __call__(self, x: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return super().__call__(x)

class BaseMetric:
    def __init__(self):
        pass
    def __call__(self, y_true:np.ndarray,y_pred:np.ndarray):
        true_y, preds = self._prepare(y_true, y_pred)
        return self._formula(true_y,preds)
    def _formula(self, y_true:np.ndarray,y_pred:np.ndarray) -> Any:
        return NotImplementedError
    def _prepare(self, y_true: np.ndarray, y_pred: np.ndarray):
        # 1. lets check if the target is binary
        binary = (y_true.shape[1] == 1) and (np.unique(y_true.size) <= 2)
        # if its, lets check if y_pred is logits or probabilities
        if binary:
            # p is the probabilities
            p = y_pred if np.all((y_pred >=0.0) & (y_pred <=1.0)) else sigmoid()(y_pred)
            preds = (p > 0.5).astype(float)
        else:# Multi class , use soft max if the values are not logits, then use argmax
            p = y_pred if np.all((y_pred >=0.0) & (y_pred <=1.0)) else softmax()(y_pred) if y_pred.shape[1] > 1 else y_pred
            preds = np.argmax(p, axis=-1, keepdims=True) if not np.all(p==y_pred) else p
        # prepare y_true, it may have the shape of (number of examples, number of classes) or (number of examples, 1) `the 1 have number of classe unique values`
        true_y = np.argmax(y_true, axis=-1, keepdims=True) if y_true.shape[1] > 1 else y_true
        return preds, true_y


class Accuracy(BaseMetric):
    def _formula(self, y_true: np.ndarray[tuple[Any, ...], np.dtype[Any]], y_pred: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> Any:
        return np.mean(y_true == y_pred)
    

class Precision(BaseMetric):
    def _formula(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
        classes = np.unique(y_true)
        precisions = []
        
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(p)
            
        return np.mean(precisions)

class Recall(BaseMetric):
    def _formula(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
        classes = np.unique(y_true)
        recalls = []
        
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(r)
            
        return np.mean(recalls)

class FBetaScore(BaseMetric):
    """
    F-Beta Score: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    F1-score is FBetaScore with beta=1.0
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta_sq = beta ** 2
        # We instantiate internal metrics to reuse their formulas
        self.precision_fn = Precision()
        self.recall_fn = Recall()

    def _formula(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any] | float:
        # Note: Precision and Recall classes will re-call _prepare internally 
        # unless we modify the call structure, so we use their _formula directly
        p = self.precision_fn._formula(y_true, y_pred)
        r = self.recall_fn._formula(y_true, y_pred)
        
        if (p + r) == 0:
            return 0.0
            
        score = (1 + self.beta_sq) * (p * r) / ((self.beta_sq * p) + r)
        return score