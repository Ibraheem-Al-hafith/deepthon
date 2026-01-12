"""
Deepthon Model Training Script: Diabetes Regression
"""

import logging
# Standard Library Imports
from typing import Dict, List, Any

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Deepthon Framework Imports
from deepthon.nn import Sequential, Layer, Dropout, BatchNorm
from deepthon.nn.optimizers import AdamW
from deepthon.nn.schedulers import StepDecay
from deepthon.nn.losses import MAE
from deepthon.pipeline import Trainer
from deepthon.utils.metrics import RMSE, RSquared
from deepthon.utils import train_test_split

# Configure standard logging to capture potential library-level logs
logging.basicConfig(level=logging.INFO, format='%(message)s')


def Regression_example() -> None:
    """
    Executes the end-to-end training pipeline for the diabetes dataset.
    
    Includes data loading, preprocessing, model architecture definition,
    training with early stopping, and visualization of results.
    
    Returns:
        None
    """

    # --- Step 1: Data Preparation ---
    X_raw, y_raw = load_diabetes(return_X_y=True)
    
    # Reshape targets for neural network compatibility (N, 1)
    X: np.ndarray = np.array(X_raw)
    y: np.ndarray = np.array(y_raw).reshape(-1, 1)
    
    # Split data into training and validation sets
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Step 2: Model Architecture ---
    # Constructing a Deep Neural Network with Batch Normalization and Dropout
    input_dim: int = X.shape[1]
    model: Sequential = Sequential([
        BatchNorm(input_dim),
        Layer(input_dim, 128, "relu"),
        BatchNorm(128),
        Layer(128, 128, "relu"),
        BatchNorm(128),
        Dropout(0.2),
        Layer(128, 1, "linear"),
    ])

    # --- Step 3: Training Configuration ---
    loss_fn: MAE = MAE()
    epochs: int = 2000
    batch_size: int = 64
    lr: float = 0.001
    
    # Optional: Exponential decay scheduler
    sch: StepDecay = StepDecay()
    
    # Optimizer initialization (Weight Decay variant of Adam)
    optimizer: AdamW = AdamW(lr=lr)
    
    # Pipeline initialization
    trainer: Trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_fn,
        batch_size=batch_size,
        early_stopping=True,
        patience=50,
        metric_fn="rmse",
        min_delta=5,
        logging_steps=0.1,
        logging="steps"
    )

    # --- Step 4: Execution & Evaluation ---
    trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    y_pred: np.ndarray = trainer.predict(X_val)
    
    metrics: Dict[str, Any] = {
        "rmse": RMSE(),
        "r2": RSquared(),
    }
    
    for key, m in metrics.items():
        score: float = m(y_val, y_pred)
        print(f"{key} :{score}")

    # --- Step 5: Visualization ---
    train_loss: List[float] = trainer.train_losses
    val_loss: List[float] = trainer.val_losses
    
    plt.figure(figsize=(12, 6))

    # Loss Curve Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss)), train_loss, label="Train")
    plt.plot(range(len(val_loss)), val_loss, label="Validation")
    plt.title("Train vs Validation Loss")
    plt.xlabel("Steps/Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Regression Scatter Plot
    plt.subplot(1, 2, 2)
    plt.scatter(x=y_pred, y=y_val, alpha=0.5)
    
    # Identity line for ideal predictions
    limits: List[float] = [
        min(float(y_val.min()), float(y_pred.min())), 
        max(float(y_val.max()), float(y_pred.max()))
    ]
    plt.plot(limits, limits, 'r--', label="Ideal")
    
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title("Scatter Plot for Diabetes Dataset")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Regression_example()