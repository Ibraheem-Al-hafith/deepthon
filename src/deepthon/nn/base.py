"""
The base class for neural network components.

This module defines the core 'Module' class, which serves as the foundational 
interface for all layers, activation functions, and models, managing state 
transitions between training and inference.

Classes:
    Module: The abstract base class for all neural network building blocks.
"""

from typing import Any, Optional, List, Dict


class Module:
    """
    Base class for all neural network modules.

    This class provides the standard interface for forward and backward passes 
    and manages the 'training' flag, which dictates behavior in layers like 
    Dropout, Batch Normalization, or Activations with caching.

    Attributes:
        training (bool): Flag indicating whether the module is in training 
            or evaluation mode. Defaults to True.

    Methods:
        __init__(): Initializes the training state.
        __call__(*args, **kwargs): Provides a functional interface for the forward pass.
        train(): Sets the training attribute to True.
        eval(): Sets the training attribute to False.
        forward(*args, **kwargs): Abstract method for forward computation logic.
        backward(*args, **kwargs): Abstract method for gradient calculation logic.
    """

    def __init__(self) -> None:
        """
        Initialize the Module. 
        
        Sets the default internal state to training mode.
        """
        self.training: bool = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Syntactic sugar to call the forward method directly via the instance.

        This allows instances to be used as functions, e.g., `output = module(input)`.

        Args:
            *args (Any): Positional arguments passed to the forward pass.
            **kwargs (Any): Keyword arguments passed to the forward pass.

        Returns:
            Any: The output of the forward method computation.
        """
        return self.forward(*args, **kwargs)

    def train(self) -> None:
        """
        Sets the module to training mode.
        
        This enables features such as gradient caching and stochastic 
        behaviors (e.g., Dropout) that are only relevant during model optimization.
        """
        self.training = True

    def eval(self) -> None:
        """
        Sets the module to evaluation mode.
        
        This disables training-only behaviors to ensure deterministic 
        outputs during inference or validation passes.
        """
        self.training = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the computation performed at every call.

        This method must be overridden by subclasses to implement the 
        specific mathematical logic of the layer or model.

        Args:
            *args (Any): Variable length argument list representing inputs.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of the forward computation (typically a tensor).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        # Placeholder logic: must be implemented by subclasses (e.g., Layer, Activation)
        raise NotImplementedError("Subclasses must implement the forward method.")

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the computation for the backward pass (gradient calculation).

        This method is responsible for computing gradients with respect to 
        inputs and internal parameters using the chain rule.

        Args:
            *args (Any): Variable length argument list (usually the upstream gradient).
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The calculated gradients for the upstream layer.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        # Placeholder logic: must be overridden for backpropagation logic
        raise NotImplementedError("Subclasses must implement the backward method.")
    def get_parameters(self) -> List[Dict[str, Any]]:
        """
        Returns the layer's weights and biases with their respective gradients.

        Returns:
            List[Dict[str, Any]]: List containing parameter, gradient, and name.
        """
        return []
    # --------------------------
    # NEW: State Serialization
    # --------------------------
    
    def get_state(self):
        """
        Returns persistent state for checkpointing.
        Default: only parameters.
        """
        state = {}
        for p in self.get_parameters():
            state[p["name"]] = p["param"]
        return state

    def load_state(self, state):
        """
        Loads persistent state from checkpoint.
        """
        for p in self.get_parameters():
            name = p["name"]
            if name in state:
                p["param"][...] = state[name]