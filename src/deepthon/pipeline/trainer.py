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
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Literal, Callable

import numpy as np

from ..utils.metrics import Accuracy, FBetaScore, Precision, Recall
from .dataloaders import DataLoader

NDArray = np.ndarray
MetricLiteral = Literal["accuracy", "f1", "precision", "recall", None]

METRICS: Dict[str, Any] = {
    "accuracy": Accuracy(),
    "f1": FBetaScore(),
    "precision": Precision(),
    "recall": Recall(),
}


class Trainer:
    """
    Modular Trainer with checkpointing, generator support, and typed design.
    Supports:
        • Raw numpy arrays or DataLoader objects
        • Save-every-N or best-on-validation checkpointing
        • Resume training safely
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_func: Any,
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

        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size

        self.metric_fn = metric_fn

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
        epoch: int,
        is_best: bool = False,
        filename: str = "checkpoint.pkl",
    ) -> None:
        if not self.checkpoint_dir:
            return

        #self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        payload = {
            "epoch": epoch,
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

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pkl"
            pickle.dump(payload, open(best_path, "wb"))

        # print(f"checkpoint saved successfully for epoch {epoch}")
    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Returns the epoch to resume from."""
        ckpt = pickle.load(open(path, "rb"))

        self.model.load_state(ckpt["model_state"])

        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            self.optimizer.load_state(ckpt["optimizer_state"])

        self.train_losses = ckpt["history"]["train_losses"]
        self.val_losses = ckpt["history"]["val_losses"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        self.start_epoch = ckpt.get("epoch", 0)
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
                        print(msg)
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # --- periodic checkpoint ---
            if self.save_every and ((epoch + 1) % self.save_every == 0):
                self.save_checkpoint(epoch+1)

            # logging
            if (epoch + 1) % log_every == 0 or (epoch + 1) == epochs:
                print(msg)


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

        for X_batch, _ in val_gen:
            outputs.append(self.model.forward(X_batch))

        outputs = np.concatenate(outputs, axis=0)
        loss = float(self.loss_func(y_val, outputs))

        return loss, outputs


    def predict(self, X: NDArray) -> NDArray:
        self.model.eval()
        return self.model.forward(X)
