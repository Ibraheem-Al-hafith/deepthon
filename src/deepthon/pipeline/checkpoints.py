import numpy as np
from pathlib import Path

def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.get_state(),
        "epoch": epoch,
        "metrics": metrics or {},
    }

    if optimizer is not None and hasattr(optimizer, "get_state"):
        checkpoint["optimizer_state"] = optimizer.get_state()

    np.save(path, **checkpoint, allow_pickle=True)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = np.load(path, allow_pickle=True).item()

    model.load_state(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state(checkpoint["optimizer_state"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
