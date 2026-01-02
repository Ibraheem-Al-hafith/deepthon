


from sklearn.datasets import make_circles, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from src.nnlib import Sequential, Layer, Dropout, BatchNorm
from src.nnlib.optimizers import Adam, SGD, AdamW
from src.nnlib.schedulers import CosineScheduler, ExponentialDecay, StepDecay
from src.nnlib.loss import BCE
from src.pipeline import Trainer
from src.utils.metrics import *
import numpy as np

def test_trainer():

    X, y = make_circles(1000, noise=0.05, random_state=42)
    # X, y = load_breast_cancer(return_X_y=True)
    # X, y = make_moons(1000, noise=0.05, random_state=42)
    y = y.reshape(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = Sequential([
        BatchNorm(2),
        Layer(2, 64, "relu"),
        BatchNorm(64),
        Dropout(0.1),
        Layer(64, 1, "sigmoid"),
    ])

    loss_fn = BCE(from_logits=False)
    epochs = 1000
    batch_size = 32
    # Calculate total steps: (Total Samples / Batch Size) * Epochs
    total_steps = (len(X_train) // batch_size) * epochs
    sch = CosineScheduler(eta_min=1e-6, max_iterations=total_steps)
    lr = 0.001
    #sch = ExponentialDecay(0.999)
    #sch = StepDecay(0.9, total_steps // 100)
    optimizer = AdamW(lr=lr, l2 = 0.01, l1=0., beta1=0.9, weigh_decay=0.01, scheduler=sch)
    # optimizer = SGD(lr=lr, l2 = 0.001, scheduler=sch)
    trainer = Trainer(
        model=model, optimizer=optimizer,loss_func=loss_fn,batch_size=batch_size
        ,early_stopping=True, patience=50,metric_fc="f1", min_delta=1e-3,logging_steps=0.1,logging="steps"
    )

    trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)
    y_pred = (trainer.predict(X_val) > 0.5).astype(int)
    metrics = {
        "accuracy": Accuracy(),
        "recall": Recall(),
        "precision": Precision(),
        "f1": FBetaScore()
    }
    for key, m in metrics.items():
        print(f"{key} :{m(y_val, y_pred)}")

    train_loss = trainer.train_losses
    val_loss = trainer.val_losses
    import matplotlib.pyplot as plt
    plt.plot(range(len(train_loss)), train_loss, label = "Train")
    plt.plot(range(len(val_loss)), val_loss, label = "Validation")
    plt.legend()
    plt.title("train vs validation loss")
    plt.show()
if __name__=="__main__":
    test_trainer()
    