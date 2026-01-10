"""
Training Orchestrator for Neural Network Models.

This module provides the 'Trainer' class, which automates the training loop, 
including batching, shuffling, backpropagation, validation, and early 
stopping logic.

Classes:
    Trainer: A high-level interface for training and evaluating models.
"""

from __future__ import annotations
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np

from ..utils.metrics import *
from ..nn.layers import Sequential
from ..nn.optimizers import BaseOptimizer
from ..nn.schedulers import BaseScheduler
from ..nn.losses import LOSS
from .dataloaders import DataLoader
import logging


# ---------------- Set up logger -------------------
logger = logging.getLogger(__name__)
_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s |"
    "%(funcName)s:%(lineno)d | %(message)s"
)
formatter = logging.Formatter(fmt=_DEFAULT_FORMAT)
ch = logging.StreamHandler()
ch.setFormatter(fmt=formatter)
logger.addHandler(ch)
# --------------------------------------------------

# NDArray = np.ndarray
MetricLiteral = Literal["accuracy", "f1", "precision", "recall", "rmse", "mse", "r2", None]

METRICS: Dict[str, Any] = {
    "accuracy": Accuracy(),
    "f1": FBetaScore(),
    "precision": Precision(),
    "recall": Recall(),
    "mse": MSE(),
    "rmse": RMSE(),
    "r2": RSquared()
}


class Trainer:
    """Orchestrates the training process for neural network models.

    This class automates the training loop, including batching, shuffling, 
    backpropagation, validation, and early stopping. It supports checkpointing
    to local storage and can handle both raw NumPy arrays and DataLoader objects.

    Attributes:
        model (Sequential): The neural network model to train.
        optimizer (BaseOptimizer): The optimization algorithm.
        loss_func (LOSS): The loss function used for training and validation.
        batch_size (int): Number of samples per training batch.
        val_batch_size (int): Number of samples per validation batch.
        metric (Optional[BaseMetric]): Metric object for evaluation.
        early_stopping (bool): Whether to stop training if validation loss plateaus.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in loss to qualify as an improvement.
        logging (str): Frequency of log output ('steps' or 'epoch').
        checkpoint_dir (Optional[Path]): Directory where checkpoints are saved.
        train_losses (List[float]): History of average training losses per epoch.
        val_losses (List[float]): History of validation losses per epoch.
        best_val_loss (float): Lowest validation loss achieved during training.
    """

    def __init__(
        self,
        model: Sequential,
        optimizer: BaseOptimizer,
        loss_func: LOSS,
        batch_size: int = 8,
        val_batch_size: Optional[int] = None,
        metric_fn: MetricLiteral = None,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 1e-4,
        logging: Literal["steps", "epoch"] = "epoch",
        logging_steps: Union[int, float] = 0.1,
        eval_steps: int = 1,
        # --- checkpoint controls ---
        checkpoint_dir: Optional[Union[str, Path]] = None,
        save_every: Optional[int] = None,       # periodic
        save_best: bool = True,                 # best val loss
    ) -> None:
        """Initializes the Trainer with model, optimizer, and training configs.

        Args:
            model: The Sequential model instance to be trained.
            optimizer: Optimizer instance (e.g., SGD, Adam).
            loss_func: Loss function instance.
            batch_size: Number of training samples per batch. Defaults to 8.
            val_batch_size: Number of validation samples per batch. 
                If None, defaults to batch_size.
            metric_fn: Literal string identifying the metric (e.g., "accuracy").
            early_stopping: If True, uses validation loss to stop training early.
            patience: Epochs to wait for improvement before early stopping.
            min_delta: Smallest improvement to reset the early stopping counter.
            logging: Frequency of logging, either "steps" or "epoch".
            logging_steps: If logging is "steps", number of steps between logs. 
                If float, interpreted as a percentage of total steps.
            eval_steps: Interval of epochs between validation runs.
            checkpoint_dir: Path to directory for saving model checkpoints.
            save_every: Frequency (in epochs) to save periodic checkpoints.
            save_best: If True, saves a 'best_model.pkl' when val_loss improves.
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size

        self.metric_fn = metric_fn
        self.metric = METRICS[str(metric_fn)] if metric_fn is not None else None

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self.logging = logging
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps

        # checkpoint settings
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_every = save_every
        self.save_best = save_best

        # histories
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # runtime training state
        self.best_val_loss: float = float("inf")
        self.stop_counter: int = 0
        self.start_epoch: int = 0  # supports resume
        self.current_epoch = 0


    # ------------------------------------------------------------------
    # Data Handling — supports ndarray OR generator
    # ------------------------------------------------------------------
    def _ensure_generator(
        self,
        X: Union[NDArray, DataLoader],
        y: Optional[NDArray],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:

        if isinstance(X, DataLoader):
            return X

        assert y is not None, "y must be provided when using raw arrays"
        return DataLoader(X, y, batch_size=batch_size, shuffle=shuffle)


    # ------------------------------------------------------------------
    # Checkpoint API
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        epoch: Optional[int] = None,
        is_best: bool = False,
        filename: str = "checkpoint.pkl",
    ) -> None:
        if not self.checkpoint_dir:
            return

        # use the provided epoch or trainer's current_epoch
        save_epoch = epoch if epoch is not None else self.current_epoch
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": save_epoch,
            "model_state": self.model.get_state(),
            "optimizer_state": getattr(self.optimizer, "get_state", lambda: None)(),
            "history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            "best_val_loss": self.best_val_loss,
        }

        path = self.checkpoint_dir / filename
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Successfully saved training configuration into :{path} - epoch: {epoch}")
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pkl"
            pickle.dump(payload, open(best_path, "wb"))
            logger.info(f"Successfully saved best model into :{best_path} - epoch: {epoch}")

        # logger.info(f"checkpoint saved successfully for epoch {epoch}")
    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Returns the epoch to resume from."""
        logger.info(f"Loading checkpoint from {path}")
        ckpt = pickle.load(open(path, "rb"))

        self.model.load_state(ckpt["model_state"])

        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            self.optimizer.load_state(ckpt["optimizer_state"])

        self.train_losses = ckpt["history"]["train_losses"]
        self.val_losses = ckpt["history"]["val_losses"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        self.start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Successfully loaded checkpoint!!")
        return self.start_epoch


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: Union[NDArray, DataLoader],
        y_train: Optional[NDArray] = None,
        X_val: Optional[Union[NDArray, DataLoader]] = None,
        y_val: Optional[NDArray] = None,
        epochs: int = 10,
        shuffle: bool = True,
    ) -> None:

        train_gen = self._ensure_generator(X_train, y_train, self.batch_size, shuffle)

        n_samples = len(train_gen)
        steps_per_epoch = math.ceil(n_samples / self.batch_size)
        total_steps = steps_per_epoch * epochs

        log_every = (
            self.logging_steps
            if isinstance(self.logging_steps, int)
            else max(1, int(self.logging_steps * (total_steps if self.logging == "steps" else epochs)))
        )

        for epoch in range(self.start_epoch, epochs):
            self.current_epoch+=1

            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_gen:
                outputs = self.model.forward(X_batch)

                loss = float(self.loss_func.forward(y_batch, outputs))
                epoch_loss += loss

                grad = self.loss_func.backward(y_batch, outputs)
                self.model.backward(grad)
                self.optimizer.step(self.model.layers)

            avg_train_loss = epoch_loss / steps_per_epoch
            self.train_losses.append(avg_train_loss)

            msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}"

            # ---------------- Validation ----------------
            if X_val is not None and y_val is not None and ((epoch + 1) % self.eval_steps == 0):
                val_loss, preds = self.validate(X_val, y_val)
                self.val_losses.append(val_loss)

                msg += f" | Val Loss: {val_loss:.4f}"

                if self.metric_fn:
                    score = METRICS[self.metric_fn](y_val, preds)
                    msg += f" | {self.metric_fn}: {score:.4f}"

                # --- Early stopping ---
                if val_loss < (self.best_val_loss - self.min_delta):
                    self.best_val_loss = val_loss
                    self.stop_counter = 0

                    if self.save_best:
                        self.save_checkpoint(epoch+1, is_best=True)

                else:
                    self.stop_counter += 1
                    if self.early_stopping and self.stop_counter >= self.patience:
                        logger.info(msg)
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            # --- periodic checkpoint ---
            if self.save_every and ((epoch + 1) % self.save_every == 0):
                self.save_checkpoint(epoch+1)

            # logging
            if (epoch + 1) % log_every == 0 or (epoch + 1) == epochs:
                logger.info(msg)


    # ------------------------------------------------------------------
    # Validation / Prediction
    # ------------------------------------------------------------------
    def validate(
        self,
        X_val: Union[NDArray, DataLoader],
        y_val: NDArray,
    ) -> Tuple[float, NDArray]:

        self.model.eval()
        outputs: List[NDArray]|NDArray = []

        val_gen = (
            self._ensure_generator(X_val, y_val, self.val_batch_size, shuffle=False)
            if isinstance(X_val, np.ndarray)
            else X_val
        )
        assert isinstance(outputs, list)
        for X_batch, _ in val_gen:
            outputs.append(self.model.forward(X_batch))
        outputs = np.concatenate(outputs, axis=0)
        
        assert isinstance(outputs, np.ndarray)
        loss:float = float(self.loss_func(y_val, outputs))

        return loss, outputs


    def predict(self, X: NDArray) -> NDArray:
        self.model.eval()
        return self.model.forward(X)