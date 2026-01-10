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
    
    state = model1.get_state()
    
    model2: Sequential = Sequential([
        Layer(2, 4, activation="relu"),
        BatchNorm(4),
        Layer(4, 8, activation="relu"),
        Dropout(),
        Layer(8, 1, activation="sigmoid")
    ])
    
    model2.load_state(state)
    for i, (layer1, layer2) in enumerate(zip(model1.layers, model2.layers)):
        if not hasattr(layer1, "get_parameters"):
            continue
        for entry1, entry2 in zip(getattr(layer1, "get_parameters")(), getattr(layer2, "get_parameters")()):
            assert np.all(entry1["param"] == entry2["param"])

    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    test_save_load_functions()