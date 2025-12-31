"""
Optimization techniques including Dropout, L1 & L2 regularization
"""

from base import Module
import numpy as np

class Dropout(Module):
    """
    Dropout is regularization method wich randomly drops a number of neurons to reduce overfitting
    """
    def __init__(self, p: float = 0.2) -> None:
        """
        Docstring for __init__
        
        :param p: dropout probability
        :type p: float
        """
        super().__init__()
        assert 0 <= p <= 1, "Drop probability should be between 0 and 1"
        self.drop_prob = p
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # If model mode is not training, return the original data
        if not self.training:
            return x
        # else, randomly drop a portion from the data
        # 1. calculate the mask
        keep_prob:float = 1 - self.drop_prob
        mask: np.ndarray = (np.random.rand(*x.shape) < keep_prob).astype(np.float32)
        # 2. calculate the output and scale the output to serve the signal strength
        output:np.ndarray = (mask * x) / keep_prob

        return output
    def backward(self, x: np.ndarray) -> np.ndarray:
        return x


def test_dropout():
    import torch
    import numpy as np
    import random
    import os

    def set_seed(seed: int = 42) -> None:

        """Sets the seed for generating random numbers in PyTorch, NumPy, and Python."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Set seed for all available GPUs
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior on CuDNN backend
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

    set_seed()
    from torch.nn import Dropout as t_Dropout
    X = np.random.randn(500, 1)
    p = 0.2
    my_dropout = Dropout(p = p)
    torch_dropout = t_Dropout(p = p)
    my_dropout.train()
    my_output = my_dropout.forward(X)
    torch_dropout.train()
    torch_output = torch_dropout(torch.from_numpy(X))
    print(f"My dropout result   : {my_output.T}")
    print(f"Torch dropout result: {torch_output.numpy().T}")
    print(f"Matches: {np.allclose(my_output, torch_output)}")

if __name__=="__main__":
    test_dropout()
