"""
Deepthon Model Training Script: MNIST Digit Classification
"""

# Standard Library Imports
import logging
from typing import Dict, List, Any, Tuple

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Deepthon Framework Imports
from deepthon.nn import Sequential, Layer, Dropout, BatchNorm
from deepthon.nn.optimizers import AdamW
from deepthon.nn.schedulers import CosineScheduler
from deepthon.nn.losses import CrossEntropy # Adjusted for multi-class
from deepthon.pipeline import Trainer
from deepthon.utils.metrics import Accuracy, Precision, Recall
from deepthon.utils import train_test_split

# Configure standard logging to capture library-level output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_mnist_trainer() -> None:
    """
    Executes the training pipeline for the MNIST dataset.
    
    The process includes:
    1. Fetching and normalizing the MNIST handwritten digit data.
    2. Reshaping and splitting data into training/validation sets.
    3. Defining a Deep Neural Network with BatchNorm and Dropout.
    4. Training with Cross-Entropy loss and Cosine Learning Rate scheduling.
    5. Evaluating performance via multi-class accuracy.
    """

    # --- Step 1: Data Preparation ---
    print("Loading MNIST dataset...")
    X_raw, y_raw = fetch_openml('mnist_784', version="active", return_X_y=True, as_frame=False, parser="liac-arff")
    
    # Normalize pixel values to [0, 1] and ensure correct float typing
    X: np.ndarray = (X_raw.astype(np.float32) / 255.0)
    # Convert labels to integers
    y: np.ndarray = np.array(y_raw).astype(int)
    
    # Split into 80% training and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Step 2: Model Architecture ---
    # Input: 784 pixels -> Hidden Layers -> Output: 10 Classes
    model: Sequential = Sequential([
        BatchNorm(784),
        Layer(784, 256, "relu"),
        BatchNorm(256),
        Dropout(0.3),
        Layer(256, 128, "relu"),
        BatchNorm(128),
        Layer(128, 10, "linear"), # Multi-class output
    ])

    # --- Step 3: Optimization & Scheduling ---
    # CrossEntropy is standard for multi-class classification
    loss_fn: CrossEntropy = CrossEntropy() 
    epochs: int = 20
    batch_size: int = 128
    lr: float = 0.001
    
    # Calculate total steps for Cosine Annealing
    steps_per_epoch: int = len(X_train) // batch_size
    total_steps: int = steps_per_epoch * epochs
    
    sch: CosineScheduler = CosineScheduler(eta_min=1e-5, max_iterations=total_steps)
    optimizer: AdamW = AdamW(lr=lr, l2=0.01, scheduler=sch)

    # --- Step 4: Training Configuration ---
    trainer: Trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_fn,
        batch_size=batch_size,
        early_stopping=True,
        patience=5,
        metric_fn="accuracy",
        logging_steps=100, # Fixed integer to ensure visibility in logs
        logging="steps"
    )

    print(f"Starting training for {epochs} epochs...")
    trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)

    # --- Step 5: Evaluation ---
    # Get class predictions (argmax of softmax output)
    y_pred_probs: np.ndarray = trainer.predict(X_val)
    y_pred: np.ndarray = np.argmax(y_pred_probs, axis=1)
    
    accuracy_metric: Accuracy = Accuracy()
    score: float|np.floating = accuracy_metric(y_val, y_pred)
    
    print("\n--- MNIST Evaluation ---")
    print(f"Validation Accuracy: {score:.4f}")

    # --- Step 6: Visualization ---
    train_loss: List[float] = trainer.train_losses
    val_loss: List[float] = trainer.val_losses
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("MNIST Training: Cross-Entropy Loss over Time")
    plt.xlabel("Logging Intervals")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    test_mnist_trainer()