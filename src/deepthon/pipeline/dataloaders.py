import numpy as np

class DataLoader:
    def __init__(self, X, y=None, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(len(X))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
        for start in range(0, len(self.X), self.batch_size):
            idx = self._indices[start:start+self.batch_size]
            Xb = self.X[idx]
            yb = self.y[idx] if self.y is not None else None
            yield Xb, yb

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
