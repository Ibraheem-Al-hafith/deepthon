import numpy as np
from deepthon.utils.metrics import Accuracy


def test_accuracy_metric():
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[1], [0], [1], [1]])

    acc = Accuracy()(y_true, y_pred)
    assert acc == 0.75
