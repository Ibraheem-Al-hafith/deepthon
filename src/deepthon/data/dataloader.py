"""
deepthon.data.dataloader
Handling Data loading logic
Classes: 
    DataLoader
"""


import numpy as np
from typing import Any, Optional

# Type alias for float-based NumPy arrays
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]

class DataLoader:
    def __init__(
            self, X: NDArray, y: Optional[NDArray] = None,
            batch_size:int=32, shuffle: bool = True):
        """
        Data loader which takes data and yield patches for efficient loading.
        Args:
            X (NDArray): features dataset to be load.
            y (Optional[NDArray]): target dataset.
            batch_size(int): number of instances per batch.
            shuffle(bool): whether to shuffle the data or not.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(len(X))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
        for start in range(0, len(self.X), self.batch_size):
            idx = self._indices[start: start+self.batch_size]
            Xb = self.X[idx]
            yb = self.y[idx] if self.y is not None else None
            yield Xb, yb
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))