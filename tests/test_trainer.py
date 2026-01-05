import numpy as np
from deepthon.nn import Layer, Sequential
from deepthon.nn.activations import ReLU, Sigmoid
from deepthon.nn.losses import BCE
from deepthon.nn.optimizers import Adam
from deepthon.pipeline import Trainer


def test_trainer_runs_single_epoch():
    X = np.random.randn(64, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    model = Sequential([
        Layer(2, 8, activation=ReLU()),
        Layer(8, 1, activation=Sigmoid()),
    ])

    trainer = Trainer(
        model=model,
        optimizer=Adam(lr=1e-3),
        loss_func=BCE(),
        metric_fn="accuracy"
    )

    trainer.train(X, y, epochs=1)
    history = trainer.train_losses

    assert len(history) == 1
