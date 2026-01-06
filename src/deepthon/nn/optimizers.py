"""
Optimizers Library for Neural Network Training.

This module provides a modular framework for parameter optimization, decoupling 
the administrative logic (state management, regularization) from the specific 
mathematical update rules (SGD, Adam, etc.).

Classes:
    BaseOptimizer: Abstract base for managing parameter updates and state.
    SGD: Stochastic Gradient Descent with momentum.
    RMSProp: Adaptive learning rate optimizer using squared gradient moving averages.
    Adam: Adaptive Moment Estimation combining momentum and RMSProp.
    AdamW: Adam with decoupled weight decay for better generalization.
"""

from typing import Any, Dict, List, Optional, Union, Protocol, Tuple
import numpy as np
from .base import Module
from .schedulers import BaseScheduler

# Type alias for float-based NumPy arrays
NDArray = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]

# =============================================================================
# BASE OPTIMIZER
# =============================================================================

class BaseOptimizer:
    """
    Foundational orchestrator for all optimization strategies.

    Manages the administrative lifecycle of training, including state tracking 
    for moments/velocity, gradient regularization (L1/L2), and coordination 
    with learning rate schedulers.

    Attributes:
        lr (float): Initial learning rate.
        l1 (float): L1 regularization factor (Lasso).
        l2 (float): L2 regularization factor (Ridge).
        scheduler (Optional[Scheduler]): Learning rate adjustment strategy.
        iterations (int): Global counter for the number of update steps taken.
        state (Dict[str, Any]): Persistent storage for optimizer-specific data 
            (e.g., velocity, moments) keyed by layer and parameter name.

    Methods:
        step(layers): Iterates through layers and applies the update rule.
        _compute_update(grad, state, current_lr): Abstract math update rule.
    """

    def __init__(
        self, 
        lr: float = 0.01, 
        l1: float = 0.0, 
        l2: float = 0.0, 
        scheduler: Optional[BaseScheduler] = None
    ) -> None:
        self.lr: float = lr
        self.l1: float = l1
        self.l2: float = l2
        self.scheduler: Optional[BaseScheduler] = scheduler
        self.iterations: int = 0
        self.state: Dict[str, Any] = {}

    def _get_lr(self) -> float:
        """Determines the current learning rate based on the scheduler."""
        if self.scheduler:
            return self.scheduler.get_lr(self.lr, self.iterations)
        return self.lr

    def _init_state(self, param: NDArray) -> Any:
        """Initializes the state buffer for a parameter (defaults to zeros)."""
        return np.zeros_like(param)

    def step(self, layers: List[Module]) -> None:
        """
        Updates model parameters based on their current gradients.

        Args:
            layers (List[Module]): A list of layers containing parameters 
                and gradients (retrieved via get_parameters()).
        """
        self.iterations += 1
        current_lr: float = self._get_lr()

        for i, layer in enumerate(layers):
            if not hasattr(layer, 'get_parameters'):
                continue

            # Iterates through weights and biases
            for entry in layer.get_parameters():
                param: NDArray = entry["param"]
                grad: NDArray = entry["grad"]
                name: str = entry["name"]
                
                # 1. State Retrieval/Initialization
                key: str = f"{i}_{name}"
                if key not in self.state:
                    self.state[key] = self._init_state(param)
                
                # 2. Apply Gradient Regularization
                total_grad: NDArray = grad.copy()
                if self.l2 > 0:
                    total_grad += self.l2 * param
                if self.l1 > 0:
                    total_grad += self.l1 * np.sign(param)

                # 3. Calculate mathematical update
                update: NDArray = self._compute_update(total_grad, self.state[key], current_lr)
                
                # 4. Apply Parameter Update
                # Decoupled Weight Decay logic for AdamW-style optimizers
                if hasattr(self, "weight_decay"):
                    wd_factor: float = getattr(self, "weight_decay")
                    param[:] -= (update + (current_lr * wd_factor * param))
                else:
                    param[:] -= update

    def _compute_update(self, grad: NDArray, state: Any, current_lr: float) -> NDArray:
        """Mathematical update logic to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement _compute_update.")
    def get_state(self):
        """Save all the optimizer attributes"""
        return self.__dict__
    def load_state(self, state: Dict[str, Any]):
        """retrieve the optimizer attributes from dictionary"""
        self.__dict__.update(state)


# =============================================================================
# OPTIMIZER IMPLEMENTATIONS
# =============================================================================

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent with Momentum.

    Navigates high-curvature regions by maintaining a velocity buffer that 
    dampens oscillations.

    Attributes:
        beta (float): Momentum coefficient (typically 0.9).
    """

    def __init__(
        self, 
        lr: float = 0.01, 
        l1: float = 0.0, 
        l2: float = 0.0, 
        beta: float = 0.9, 
        scheduler: Optional[BaseScheduler] = None
    ) -> None:
        super().__init__(lr, l1, l2, scheduler)
        self.beta: float = beta

    def _compute_update(self, grad: NDArray, state: NDArray, current_lr: float) -> NDArray:
        """Update: v = beta * v + grad; return lr * v"""
        state[:] = self.beta * state + grad
        return current_lr * state


class RMSProp(BaseOptimizer):
    """
    Root Mean Square Propagation.

    Adaptive learning rate optimizer that normalizes gradients by the 
    root mean square of their moving average.
    """

    def __init__(
        self, 
        lr: float = 0.01, 
        rho: float = 0.99, 
        epsilon: float = 1e-7, 
        l1: float = 0.0, 
        l2: float = 0.0, 
        scheduler: Optional[BaseScheduler] = None
    ) -> None:
        super().__init__(lr, l1, l2, scheduler)
        self.rho: float = rho
        self.epsilon: float = epsilon

    def _compute_update(self, grad: NDArray, state: NDArray, current_lr: float) -> NDArray:
        """Update: s = rho * s + (1-rho) * grad^2; return (lr * grad) / sqrt(s + eps)"""
        state[:] = self.rho * state + (1 - self.rho) * np.square(grad)
        return (current_lr * grad) / (np.sqrt(state) + self.epsilon)


class Adam(BaseOptimizer):
    """
    Adaptive Moment Estimation.

    Combines momentum and adaptive scaling with bias correction for 
    improved stability in early training stages.
    """

    def __init__(
        self, 
        lr: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-7, 
        l1: float = 0.0, 
        l2: float = 0.0, 
        scheduler: Optional[BaseScheduler] = None
    ) -> None:
        super().__init__(lr, l1, l2, scheduler)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

    def _init_state(self, param: NDArray) -> Dict[str, NDArray]:
        """Initializes both first (m) and second (v) moments."""
        return {
            "m": np.zeros_like(param), 
            "v": np.zeros_like(param)
        }

    def _compute_update(self, grad: NDArray, state: Dict[str, NDArray], current_lr: float) -> NDArray:
        # 1. Update moments
        state["m"][:] = self.beta1 * state["m"] + (1 - self.beta1) * grad
        state["v"][:] = self.beta2 * state["v"] + (1 - self.beta2) * np.square(grad)

        # 2. Apply bias correction to counteract zero-initialization
        m_hat: NDArray = state["m"] / (1 - self.beta1 ** self.iterations)
        v_hat: NDArray = state["v"] / (1 - self.beta2 ** self.iterations)
        
        # 3. Calculate adaptive update
        return (current_lr * m_hat) / (np.sqrt(v_hat) + self.epsilon)


class AdamW(Adam):
    """
    Adam with Decoupled Weight Decay.

    Preferred for modern architectures like Transformers. It decouples weight 
    decay from the adaptive gradient scaling to ensure regularization remains 
    effective.
    """

    def __init__(
        self, 
        lr: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-7, 
        l1: float = 0.0, 
        l2: float = 0.0, 
        scheduler: Optional[BaseScheduler] = None, 
        weight_decay: float = 0.01
    ) -> None:
        super().__init__(lr, beta1, beta2, epsilon, l1, l2, scheduler)
        self.weight_decay: float = weight_decay