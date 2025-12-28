"""The base class for neural network components.
It's meant to handle the training and evaluation modes.
"""

class Module:
    """Base class for all neural network modules.
    
    This class provides methods to switch between training and evaluation modes.
    """

    def __init__(self):
        self.training = True

    def train(self):
        """Sets the module to training mode."""
        self.training = True

    def eval(self):
        """Sets the module to evaluation mode."""
        self.training = False