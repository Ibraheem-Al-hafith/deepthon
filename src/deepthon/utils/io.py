import numpy as np
from numpy.typing import NDArray
from typing import Dict
from deepthon.nn import Sequential

def save_model(model: Sequential, path: str) -> None:
    """
    Save model weights to a compressed NumPy archive
    Args:
        model (Sequential): model to be saved
        path (str): save path
    """
    params: Dict[str, NDArray] = {}
    for idx, layer in enumerate(model.layers):
        if not hasattr(layer, "get_parameters"):
            continue
        for entry in getattr(layer, "get_parameters")():
            params[f"layer_{idx}_{entry["name"]}"] = entry["param"]
    
    np.savez_compressed(path, **params, allow_pickle=True)

def load_model(model: Sequential, path: str) -> Sequential:
    """
    Load weights into an existing model structure.
    the architecture must match the saved model.
    Args:
        model (Sequential): model template
        path (str): saved model path
    """

    params = np.load(path)

    for idx, layer in enumerate(model.layers):
        if not hasattr(layer, "get_parameters"):
            continue
        for entry in getattr(layer, "get_parameters")():
            setattr(layer, entry["name"], params[f"layer_{idx}_{entry["name"]}"])
    return model