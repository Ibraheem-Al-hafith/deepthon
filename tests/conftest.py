import numpy as np
import pytest


@pytest.fixture
def random_data():
    X = np.random.randn(32, 3)
    y = (X.sum(axis=1) > 0).astype(int).reshape(-1, 1)
    return X, y
