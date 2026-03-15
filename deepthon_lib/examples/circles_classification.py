"""
Deepthon Model Training Script: Binary Classification on Circles Dataset
"""

# Standard Library Imports
import logging
from typing import Dict, List, Any, Tuple

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Deepthon Framework Imports
from deepthon.nn import Sequential, Layer, Dropout, BatchNorm
from deepthon.nn.optimizers import Adam, AdamW, SGD
from deepthon.nn.schedulers import CosineScheduler, ExponentialDecay, StepDecay
from deepthon.nn.losses import BCE
from deepthon.pipeline import Trainer
from deepthon.utils.metrics import Accuracy, Recall, FBetaScore, Precision
from deepthon.utils import train_test_split

# Configure standard logging to capture potential library-level logs
logging.basicConfig(level=logging.INFO, format='%(message)s')

def Classification_example() -> None:
    """
    Tests the neural network trainer using a synthetic circular dataset.

    This function performs the following steps:
    1. Generates and preprocesses the 'make_circles' dataset.
    2. Constructs a Sequential model with BatchNorm and Dropout.
    3. Configures the AdamW optimizer with a Cosine Learning Rate Scheduler.
    4. Executes training with early stopping and step-based logging.
    5. Evaluates and prints classification metrics.
    6. Plots the training and validation loss history.

    Returns:
        None
    """

    # --- Step 1: Data Preparation ---
    X: np.ndarray
    y: np.ndarray
    X, y = make_circles(n_samples=1000, noise=0.05, random_state=42)
    
    # Reshape labels for binary cross-entropy compatibility (N, 1)
    y = y.reshape(-1, 1)
    
    # Stratified split to maintain class balance in both sets
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Step 2: Model Architecture ---
    # Designed for non-linear separation in 2D space
    model: Sequential = Sequential([
        BatchNorm(2),
        Layer(2, 64, "relu"),
        BatchNorm(64),
        Dropout(0.2),
        Layer(64, 1, "sigmoid"),
    ])

    # --- Step 3: Optimization & Scheduling ---
    loss_fn: BCE = BCE(from_logits=False)
    epochs: int = 2000
    batch_size: int = 64
    lr: float = 0.001
    
    # Iterations per epoch = Total Samples / Batch Size
    # Total steps = iterations_per_epoch * total_epochs
    total_steps: int = (len(X_train) // batch_size) * epochs
    
    # Cosine Annealing scheduler reduces LR over the course of training
    sch: CosineScheduler = CosineScheduler(eta_min=1e-6, max_iterations=total_steps)
    
    # AdamW with weight decay (l2=0.01)
    optimizer: AdamW = AdamW(lr=lr, l2=0.01, scheduler=sch)

    # --- Step 4: Training Configuration ---
    # Note: If no output appears, try lowering logging_steps to a fixed integer (e.g., 10)
    trainer: Trainer = Trainer(
        model=model, 
        optimizer=optimizer,
        loss_func=loss_fn,
        batch_size=batch_size,
        early_stopping=True, 
        patience=50,
        metric_fn="f1", 
        min_delta=1e-3,
        logging_steps=0.1,  # Logs every 10% of total steps
        logging="steps"
    )

    print("Starting training...")
    trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)
    print("Training complete.")

    # --- Step 5: Evaluation ---
    # Thresholding probabilities at 0.5 for binary classification
    raw_output: np.ndarray = trainer.predict(X_val)
    y_pred: np.ndarray = (raw_output > 0.5).astype(int)
    
    metrics: Dict[str, Any] = {
        "accuracy": Accuracy(),
        "recall": Recall(),
        "precision": Precision(),
        "f1": FBetaScore()
    }
    
    print("\n--- Validation Metrics ---")
    for name, metric_obj in metrics.items():
        score: float = metric_obj(y_val, y_pred)
        print(f"{name.capitalize():<10}: {score:.4f}")

    # --- Step 6: Visualization ---
    train_loss: List[float] = trainer.train_losses
    val_loss: List[float] = trainer.val_losses
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Logging Intervals")
    plt.ylabel("BCE Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    Classification_example()