import numpy as np
from deepthon.nn import Layer, Sequential
from deepthon.nn.activations import ReLU, Sigmoid
from deepthon.nn.losses import BCE
from deepthon.nn.optimizers import Adam
from deepthon.pipeline import Trainer
from deepthon.nn.schedulers import CosineScheduler
from pathlib import Path
import shutil


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
def test_trainer_checkpoinging():
    X = np.random.randn(64, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    model = Sequential([
        Layer(2, 8, activation=ReLU()),
        Layer(8, 1, activation=Sigmoid()),
    ])
    chk_dir = Path("./tests/chkpts")
    #chk_dir.mkdir(exist_ok=True)
    trainer = Trainer(
        model=model,
        optimizer=Adam(lr=1e-3, scheduler=CosineScheduler(1e-6, 10)),
        loss_func=BCE(),
        metric_fn="accuracy",save_every=1,
        checkpoint_dir=chk_dir
    )

    trainer.train(X, y, epochs=5)
    st = trainer.load_checkpoint(chk_dir / "checkpoint.pkl")
    trainer.train(X, y,epochs=10)
    history = trainer.train_losses
    shutil.rmtree(chk_dir)
    assert len(history) == 10,f"{len(history)} != 10"