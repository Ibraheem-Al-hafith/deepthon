import numpy as np
from ..utils.metrics import Accuracy, FBetaScore, Recall, Precision
from typing import Literal

metrics = {
    "accuracy": Accuracy(),
    "f1": FBetaScore(),
    "precision": Precision(),
    "recall": Recall()
}
class Trainer:
    def __init__(
            self, model, optimizer, loss_func,
            batch_size = 8,early_stopping = False,
            patience = 5, min_delta = 1e-4,
            val_batch_size = None, metric_fc: Literal["accuracy", "f1", "precision", "recall", None] = None,
            logging: Literal["steps","epoch"] = "epoch", logging_steps: int | float = 0.1, eval_steps:int = 1
            ) -> None:
        """
        Trainer class for handling training loop
        
        :param model: Neural Network model to be trained
        :param optimizer: training optimizer
        :param loss_func: training loss function
        :param early_stopping: whether to stop training after
        a certain steps if validation metrics didn't improve
        :param patience: number of early stopping steps
        :param min_delta: the min difference between validation metrics
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.metric_fn = metric_fc
        self.logging =  logging
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps

        self.train_losses = []
        self.val_losses = []
    
    def train(
            self, X_train, y_train, X_val = None, y_val = None, 
            epochs = 10, shuffle = True,
            ):
        n_samples = X_train.shape[0]
        total_steps = (self.batch_size // n_samples + 1) * epochs
        logging_steps = self.logging_steps
        if isinstance(logging_steps, float):logging_steps *= total_steps

        for epoch in range(epochs):
            # 1. shuffle the data at the start of each epoch
            indices = np.random.permutation(n_samples) if shuffle else range(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            counter = 0
            best_val_loss = np.inf
            # ---- training loop -----
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i: i + self.batch_size]
                y_batch = y_shuffled[i: i + self.batch_size]

                # forward pass
                output = self.model.forward(X_batch)

                # compute the loss
                batch_loss = self.loss_func.forward(y_batch, output)
                epoch_loss += batch_loss

                # backward pass
                grad = self.loss_func.backward(y_batch, output)
                self.model.backward(grad)

                # update the parameters
                self.optimizer.step(self.model.layers)
            self.train_losses.append(epoch_loss / (n_samples // self.batch_size))

            # ---- validation loop ----
            if (((X_val is not None) and (epoch+1 % self.eval_steps == 0)) or self.early_stopping):
                val_loss, preds = self.validate(X_val, y_val)
                self.val_losses.append(val_loss)
                msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {self.train_losses[-1]:.4f} - Val Loss: {val_loss:.4f}"
                if self.metric_fn is not None:
                    msg+=f"\n {self.metric_fn}: {metrics[self.metric_fn](y_val, preds)}"
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter+=1
                    if counter >= self.patience:
                        print(f"No Improvement for {self.patience} steps, stop training")
                        break
            else:
                msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {self.train_losses[-1]:.4f}"
            
            if ((((epoch+1) % logging_steps) == 0) or (epoch+1 == epochs)):
                print(msg)

            
    def validate(self, X_val, y_val):
        self.model.eval()
        output = self.model(X_val)
        return self.loss_func.forward(y_val, output), output
    def predict(self, X):
        self.model.eval()
        return self.model(X)
