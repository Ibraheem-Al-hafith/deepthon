import numpy as np
import math
from typing import Literal
from ..utils.metrics import *

metrics = {
    "accuracy": Accuracy(),
    "f1": FBetaScore(),
    "precision": Precision(),
    "recall": Recall()
}

class Trainer:
    def __init__(
            self, model, optimizer, loss_func,
            batch_size=8, early_stopping=False,
            patience=5, min_delta=1e-4,
            val_batch_size=None, metric_fc: Literal["accuracy", "f1", "precision", "recall", None] = None,
            logging: Literal["steps", "epoch"] = "epoch", logging_steps: int | float = 0.1, eval_steps: int = 1
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.metric_fn = metric_fc
        self.logging = logging
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps

        self.train_losses = []
        self.val_losses = []

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, shuffle=True):
        n_samples = X_train.shape[0]
        steps_per_epoch = math.ceil(n_samples / self.batch_size)
        total_steps = steps_per_epoch * epochs
        
        log_every_n = self.logging_steps
        if isinstance(log_every_n, float):
            log_every_n = max(1, int(log_every_n * (total_steps if self.logging == "steps" else epochs)))

        # PERSISTENT Early Stopping variables
        best_val_loss = np.inf
        counter = 0

        for epoch in range(epochs):
            self.model.train() # Set model to training mode
            indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                # Forward pass
                output = self.model.forward(X_batch)
                batch_loss = self.loss_func.forward(y_batch, output)
                epoch_loss += batch_loss

                # Backward pass & Update
                grad = self.loss_func.backward(y_batch, output)
                self.model.backward(grad)
                self.optimizer.step(self.model.layers)

            avg_train_loss = epoch_loss / steps_per_epoch
            self.train_losses.append(avg_train_loss)

            # Validation logic
            do_eval = (X_val is not None) and ((epoch + 1) % self.eval_steps == 0)
            
            if do_eval:
                val_loss, preds = self.validate(X_val, y_val)
                self.val_losses.append(val_loss)
                
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
                if self.metric_fn:
                    score = metrics[self.metric_fn](y_val, preds)
                    msg += f" | {self.metric_fn}: {score:.4f}"

                # Check Early Stopping
                if val_loss < (best_val_loss - self.min_delta):
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                
                if self.early_stopping and counter >= self.patience:
                    print(msg)
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}"

            if (epoch + 1) % log_every_n == 0 or (epoch + 1) == epochs:
                print(msg)

    def validate(self, X_val, y_val):
        self.model.eval()
        # Use batching for validation too if X_val is large
        output = self.model.forward(X_val) 
        loss = self.loss_func.forward(y_val, output)
        return loss, output
    def predict(self, X):
        self.model.eval()
        return self.model(X)