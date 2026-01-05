"""
Training Orchestrator for Neural Network Models.

This module provides the 'Trainer' class, which automates the training loop, 
including batching, shuffling, backpropagation, validation, and early 
stopping logic.

Classes:
    Trainer: A high-level interface for training and evaluating models.
"""

import numpy as np
import math
from typing import Literal, Dict, List, Optional, Union, Any, Tuple, Callable
from ..utils.metrics import Accuracy, FBetaScore, Precision, Recall

# Type Aliases for clarity
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]
MetricLiteral = Literal["accuracy", "f1", "precision", "recall", None]

# Global metrics registry
METRICS: Dict[str, Any] = {
    "accuracy": Accuracy(),
    "f1": FBetaScore(),
    "precision": Precision(),
    "recall": Recall()
}


class Trainer:
    """
    Orchestrator for model training and evaluation.

    The Trainer handles the iterative process of feeding batches to a model, 
    calculating losses, updating parameters via an optimizer, and monitoring 
    performance on validation data.

    Attributes:
        model (Any): The neural network model (expected to inherit from Module).
        optimizer (Any): The optimization algorithm (e.g., SGD, Adam).
        loss_func (Any): The loss function instance.
        batch_size (int): Number of samples per training update.
        early_stopping (bool): Whether to stop training if validation loss plateaus.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        val_batch_size (int): Number of samples per validation forward pass.
        metric_fn (MetricLiteral): Key for the metric used during evaluation.
        logging (Literal["steps", "epoch"]): Frequency type for console output.
        logging_steps (Union[int, float]): Steps or percentage interval for logging.
        eval_steps (int): Frequency of validation passes (in epochs).
        train_losses (List[float]): History of average training losses per epoch.
        val_losses (List[float]): History of validation losses.

    Methods:
        train(X_train, y_train, X_val, y_val, epochs, shuffle): Executes the training loop.
        validate(X_val, y_val): Computes loss and predictions on validation data.
        predict(X): Generates model predictions in evaluation mode.
    """

    def __init__(
        self, 
        model: Any, 
        optimizer: Any, 
        loss_func: Any,
        batch_size: int = 8, 
        early_stopping: bool = False,
        patience: int = 5, 
        min_delta: float = 1e-4,
        val_batch_size: Optional[int] = None, 
        metric_fn: MetricLiteral = None,
        logging: Literal["steps", "epoch"] = "epoch", 
        logging_steps: Union[int, float] = 0.1, 
        eval_steps: int = 1
    ) -> None:
        """
        Initializes the Trainer with model components and hyperparameters.
        """
        self.model: Any = model
        self.optimizer: Any = optimizer
        self.loss_func: Any = loss_func
        self.early_stopping: bool = early_stopping
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.batch_size: int = batch_size
        self.val_batch_size: int = val_batch_size or batch_size
        self.metric_fn: MetricLiteral = metric_fn
        self.logging: str = logging
        self.logging_steps: Union[int, float] = logging_steps
        self.eval_steps: int = eval_steps

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train(
        self, 
        X_train: NDArray, 
        y_train: NDArray, 
        X_val: Optional[NDArray] = None, 
        y_val: Optional[NDArray] = None, 
        epochs: int = 10, 
        shuffle: bool = True
    ) -> None:
        """
        Executes the full training lifecycle over multiple epochs.

        Args:
            X_train (NDArray): Training feature matrix.
            y_train (NDArray): Training target labels.
            X_val (Optional[NDArray]): Validation feature matrix.
            y_val (Optional[NDArray]): Validation labels.
            epochs (int): Number of times to iterate over the dataset.
            shuffle (bool): Whether to re-order samples every epoch.
        """
        n_samples: int = X_train.shape[0]
        steps_per_epoch: int = math.ceil(n_samples / self.batch_size)
        total_steps: int = steps_per_epoch * epochs
        
        # Calculate logging interval
        log_every_n: int = self.logging_steps if isinstance(self.logging_steps, int) else \
            max(1, int(self.logging_steps * (total_steps if self.logging == "steps" else epochs)))

        # State tracking for Early Stopping
        best_val_loss: float = np.inf
        stop_counter: int = 0

        

        for epoch in range(epochs):
            self.model.train()  # Activate training mode (e.g., enable Dropout)
            
            # 1. Shuffling logic
            indices: NDArray | np.typing.NDArray[Any]= np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
            X_shuffled: NDArray = X_train[indices]
            y_shuffled: NDArray = y_train[indices]

            epoch_loss: float = 0.0
            
            # 2. Mini-batch Iteration
            for i in range(0, n_samples, self.batch_size):
                X_batch: NDArray = X_shuffled[i : i + self.batch_size]
                y_batch: NDArray = y_shuffled[i : i + self.batch_size]

                # Forward pass: Compute predictions and loss
                output: NDArray = self.model.forward(X_batch)
                batch_loss: float = float(self.loss_func.forward(y_batch, output))
                epoch_loss += batch_loss

                # Backward pass: Compute gradients and update weights
                grad: NDArray = self.loss_func.backward(y_batch, output)
                self.model.backward(grad)
                self.optimizer.step(self.model.layers)

            avg_train_loss: float = epoch_loss / steps_per_epoch
            self.train_losses.append(avg_train_loss)

            # 3. Validation and Performance Monitoring
            do_eval: bool = (X_val is not None and y_val is not None) and ((epoch + 1) % self.eval_steps == 0)
            
            if do_eval:
                assert X_val is not None
                assert y_val is not None
                val_loss, preds = self.validate(X_val = X_val, y_val = y_val)
                self.val_losses.append(val_loss)
                
                msg: str = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
                
                # Compute optional metrics (Accuracy, F1, etc.)
                if self.metric_fn:
                    score: float = METRICS[self.metric_fn](y_val, preds)
                    msg += f" | {self.metric_fn}: {score:.4f}"

                # 4. Early Stopping Logic
                if val_loss < (best_val_loss - self.min_delta):
                    best_val_loss = val_loss
                    stop_counter = 0
                else:
                    stop_counter += 1
                
                if self.early_stopping and stop_counter >= self.patience:
                    print(msg)
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}"

            # 5. Logging
            if (epoch + 1) % log_every_n == 0 or (epoch + 1) == epochs:
                print(msg)

    def validate(self, X_val: NDArray, y_val: NDArray) -> Tuple[float, NDArray]:
        """
        Performs evaluation on a validation dataset.

        Args:
            X_val (NDArray): Validation features.
            y_val (NDArray): Validation targets.

        Returns:
            Tuple[float, NDArray]: A tuple of (scalar loss, predictions).
        """
        self.model.eval()  # Deactivate training behavior
        output: NDArray = np.empty_like(X_val)
        # Iterate through X_val
        for i in range(0, len(X_val), self.val_batch_size):
            output[i: i+self.val_batch_size] = self.model.forward(X_val[i: i+self.val_batch_size]) 
        # Calculate the loss
        loss: float = float(self.loss_func(y_val, output))
        
        return loss, output

    def predict(self, X: NDArray) -> NDArray:
        """
        Generates predictions for the given input data.

        Args:
            X (NDArray): Input feature matrix.

        Returns:
            NDArray: Model predictions.
        """
        self.model.eval()
        return self.model(X)