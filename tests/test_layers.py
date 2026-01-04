import numpy as np
from deepthon.nn import Layer, Sequential
from deepthon.nn.activations import ReLU, Sigmoid


def test_layer_forward_backward_shapes():
    x = np.random.randn(8, 4)
    layer = Layer(4, 3)

    y = layer.forward(x)
    grad = layer.backward(np.ones_like(y))

    assert y.shape == (8, 3)
    assert grad.shape == (8, 4)


def test_sequential_forward_backward_chain():
    x = np.random.randn(5, 2)

    model = Sequential([
        Layer(2, 4, activation=ReLU()),
        Layer(4, 1, activation=Sigmoid()),
    ])

    y = model.forward(x)
    grad = model.backward(np.ones_like(y))

    assert y.shape == (5, 1)
    assert grad.shape == (5, 2)
