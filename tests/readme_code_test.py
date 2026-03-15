import numpy as np
from deepthon.nn import Sequential, Layer
from deepthon.nn.activations import ReLU, Sigmoid
from deepthon.nn.losses import BCE
from deepthon.nn.optimizers import Adam
from deepthon.pipeline import Trainer

def test_readme_code():
    X = np.random.randn(500, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    model = Sequential([
        Layer(2, 16, activation=ReLU()),
        Layer(16, 8, activation=ReLU()),
        Layer(8, 1, activation=Sigmoid()),
    ])
    
    optimizer = Adam(lr=1e-3)
    loss = BCE()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss,
        batch_size=32,
        metric_fn="accuracy",
    )
    
    trainer.train(X, y, epochs=20)