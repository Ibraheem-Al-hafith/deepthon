"""
Learning Rate Scheduler Library.

This module provides a collection of learning rate decay strategies used in 
machine learning optimization. These strategies help in fine-tuning the 
step size during the training process to achieve better convergence.

Classes:
    BaseScheduler: Abstract interface for all scheduler types.
    ExponentialDecay: Gradually reduces LR by a factor every iteration.
    StepDecay: Reduces LR by a factor after a fixed number of steps.
    CosineScheduler: Follows a cosine curve for annealing LR.
"""

from typing import Union
import numpy as np


# =============================================================================
# BASE SCHEDULER INTERFACE
# =============================================================================

class BaseScheduler:
    """
    Abstract base class for all learning rate scheduler implementations.
    
    Provides the standard interface for calculating decayed learning rates
    based on the current training iteration and the initial rate.

    Attributes:
        None

    Methods:
        get_lr(initial_lr, iteration): Abstract method to calculate current LR.
    """

    def get_lr(self, initial_lr: float, iteration: int) -> float:
        """
        Calculate the learning rate for the given iteration.

        Args:
            initial_lr (float): The starting learning rate at iteration 0.
            iteration (int): The current training step/iteration index.

        Returns:
            float: The calculated learning rate.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement get_lr method.")


# =============================================================================
# SCHEDULER IMPLEMENTATIONS
# =============================================================================

class ExponentialDecay(BaseScheduler):
    """
    Exponential Decay Scheduler.

    Reduces the learning rate at every iteration by multiplying it by 
    a decay rate.
    Formula: lr = initial_lr * (decay_rate ^ iteration)

    Attributes:
        decay_rate (float): The factor by which the LR is multiplied 
            each step (e.g., 0.99).

    Methods:
        get_lr(initial_lr, iteration): Computes exponential decay.
    """

    def __init__(self, decay_rate: float = 0.96) -> None:
        """
        Args:
            decay_rate (float): Decay factor. Defaults to 0.96.
        """
        self.decay_rate: float = decay_rate

    def get_lr(self, initial_lr: float, iteration: int) -> float:
        """
        Calculate the exponentially decayed learning rate.

        Args:
            initial_lr (float): The starting learning rate.
            iteration (int): The current iteration index.

        Returns:
            float: The decayed learning rate.
        """
        return initial_lr * (self.decay_rate ** iteration)


class StepDecay(BaseScheduler):
    """
    Step Decay Scheduler.

    Reduces the learning rate by a factor after a specific number of 
    iterations (a "step").

    Attributes:
        decay_rate (float): The factor to multiply the LR by at each step.
        step_size (int): The number of iterations between each decay event.

    Methods:
        get_lr(initial_lr, iteration): Computes step-based decay.
    """

    def __init__(self, decay_rate: float = 0.1, step_size: int = 100) -> None:
        """
        Args:
            decay_rate (float): Multiplicative factor for decay.
            step_size (int): Interval of iterations for decay.
        """
        self.decay_rate: float = decay_rate
        self.step_size: int = step_size

    def get_lr(self, initial_lr: float, iteration: int) -> float:
        """
        Calculate the step-decayed learning rate.

        Args:
            initial_lr (float): The starting learning rate.
            iteration (int): The current iteration index.

        Returns:
            float: The step-decayed learning rate.
        """
        # Determine how many 'steps' have passed using integer division
        step: int = iteration // self.step_size
        return initial_lr * (self.decay_rate ** step)


class CosineScheduler(BaseScheduler):
    """
    Cosine Annealing Scheduler.

    Decreases the learning rate following a cosine curve toward a 
    minimum value (eta_min) over a fixed number of iterations. 
    This is often used to "warm down" the model at the end of training.

    

    Attributes:
        eta_min (float): Minimum learning rate value (the floor).
        max_iterations (int): Total number of iterations for the decay cycle.

    Methods:
        get_lr(initial_lr, iteration): Computes cosine-based annealing.
    """

    def __init__(self, eta_min: float, max_iterations: int) -> None:
        """
        Args:
            eta_min (float): The lowest value the LR can reach.
            max_iterations (int): Total training steps expected.
        """
        self.eta_min: float = eta_min
        self.max_iterations: int = max_iterations

    def get_lr(self, initial_lr: float, iteration: int) -> float:
        """
        Calculate the learning rate using cosine annealing.

        Args:
            initial_lr (float): The starting learning rate.
            iteration (int): The current iteration index.

        Returns:
            float: The learning rate following the cosine path.
        """
        # Ensure the progress fraction does not exceed 1.0
        fraction: float = min(iteration / self.max_iterations, 1.0)
        
        # Apply cosine formula: 
        # lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * iteration / max_iter))
        lr: float = self.eta_min + (initial_lr - self.eta_min) * (
            0.5 * (1.0 + np.cos(np.pi * fraction))
        )
        return lr