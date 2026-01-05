import numpy as np

class DataGenerator:
    """Data generator that yields data batches in and efficient manner"""
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        """
        Args:
            x (np.ndarray): data, usually the features data.
            y (np.ndarray): target data.
            batch_size (int): the number of instences returned at each iteration
            shuffle (bool): whether to shuffle the data on the fly or not (recommended)
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))

    def _shuffle_data(self):
        """Randomizes the order of indices for a new epoch."""
        np.random.shuffle(self.indices)

    def get_batches(self):
        """Yields batches of (x, y) until the end of the data."""
        if self.shuffle:
            self._shuffle_data()
            
            # If i + batch_size > len(x), Python slicing safely returns 
            # the remaining elements.
        for i in range(0, len(self.x), self.batch_size):
            batch_idx = self.indices[i : i + self.batch_size]

            # Grab the batch
            batch_x = self.x[batch_idx]
            batch_y = self.y[batch_idx]

            # Flatten on the fly: (Batch, 28, 28) -> (Batch, 784)
            batch_x_flat = batch_x.reshape(len(batch_x), -1)

            yield batch_x_flat, batch_y