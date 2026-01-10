import numpy as np
from typing import Tuple, Optional, Generator, Any

class DataLoader:
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray]=None, batch_size: int=32, shuffle: bool=True):
        """
        Make an efficient data loader for batching processing
        Args:
            X (np.ndarray): features data.
            y (Optional[np.ndarray]): optional target data.
            batch_size (int): number of instances to be returend at each iteration.
            shuffle (bool): whethet to shuffle the data or not
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(len(X))

    def __iter__(self) -> Any:
        if self.shuffle:
            np.random.shuffle(self._indices)
        for start in range(0, len(self.X), self.batch_size):
            idx = self._indices[start:start+self.batch_size]
            Xb = self.X[idx]
            yb = self.y[idx] if self.y is not None else None
            yield Xb, yb

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
