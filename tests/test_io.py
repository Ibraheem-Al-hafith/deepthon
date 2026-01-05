from deepthon.utils.io import save_model, load_model
from deepthon.nn import Sequential, Layer
from deepthon.nn.layers import Dropout, BatchNorm
import numpy as np
import os



def test_save_load_functions():
    model1: Sequential = Sequential([
        Layer(2, 4, activation="relu"),
        BatchNorm(4),
        Layer(4, 8, activation="relu"),
        Dropout(),
        Layer(8, 1, activation="sigmoid")
    ])
    
    path: str = "./tests/model.npz"
    
    save_model(model1, path=path)
    
    model: Sequential = Sequential([
        Layer(2, 4, activation="relu"),
        BatchNorm(4),
        Layer(4, 8, activation="relu"),
        Dropout(),
        Layer(8, 1, activation="sigmoid")
    ])
    
    model2 = load_model(model, path=path)
    for i, (layer1, layer2) in enumerate(zip(model1.layers, model2.layers)):
        if not hasattr(layer1, "get_parameters"):
            continue
        for entry1, entry2 in zip(getattr(layer1, "get_parameters")(), getattr(layer2, "get_parameters")()):
            assert np.all(entry1["param"] == entry2["param"])

    if os.path.exists(path):
        os.remove(path)