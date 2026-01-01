import numpy as np
from .schedulers import *
"""
### Phase 1: The BaseOptimizer (Parent) 🏛️
This class manages the "administrative" tasks.
1. **Initialize**:
    * Store hyper-parameters (lr,l1 ,l2 ).
    * Initialize an empty dictionary `self.state` to store velocity/buffers.
    * Initialize `self.iterations = 0` to track the number of steps.
2. **Learning Rate Management**:
    * Create a helper method (`_get_lr`) that checks if a scheduler exists. If yes, it calculates the new rate; if no, it returns the base .
3. **The Step Loop**:
    * Increment `self.iterations`.
    * Get the `current_lr`.
    * **Loop** through every layer in the model.
    * **Loop** through every parameter (weight/bias) in that layer.
4. **Parameter Preparation**:
    * Generate a **Unique Key** (e.g., `layer_index_param_name`).
    * If the key isn't in `self.state`, create a zero-filled array of the same shape as the parameter. 🏁
    * **Regularize**: Calculate `total_grad = grad + (self.l2 * param) + (self.l1 * sign(param))`. ⚖️
5. **Execution**:
    * Call the child's `_compute_update` method, passing the `total_grad`, the `state` (velocity) for that key, and the `current_lr`.
    * Subtract the returned update from the parameter **in-place** (`param -= update`).
"""
class Optimizer:
    def __init__(self, lr=0.01, l1=0.0, l2=0.0, scheduler=None):
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.scheduler = scheduler
        self.iterations = 0
        self.state = {} # Stores velocity, etc.

    def _get_lr(self):
        # If a scheduler exists, use it; otherwise, use the base lr
        if self.scheduler:
            return self.scheduler.get_lr(self.lr, self.iterations)
        return self.lr
    def _init_state(self, param):
        """Default state is just a zero array (for SGD, RMSProp)"""
        return np.zeros_like(param)

    def step(self, layers):
        self.iterations += 1
        current_lr = self._get_lr()

        for i, layer in enumerate(layers):
            if not hasattr(layer, 'get_parameters'):
                continue

            for entry in layer.get_parameters():
                param = entry["param"]
                grad = entry["grad"]
                name = entry["name"]
                
                # 1. Create a unique key for the state dictionary
                key = f"{i}_{name}"
                if key not in self.state:
                    self.state[key] = self._init_state(param)
                
                # 2. Add Regularization to the gradient
                total_grad = grad.copy()
                if self.l2 > 0:
                    total_grad += self.l2 * param
                if self.l1 > 0:
                    total_grad += self.l1 * np.sign(param)

                # 3. Call the child's specific math
                update = self._compute_update(total_grad, self.state[key], current_lr)
                
                # 4. Apply the update
                if hasattr(self, "weight_decay"):
                    param[:] -= update + (current_lr * self.__getattribute__("weight_decay") * param)
                else:
                    param[:] -= update
    def _compute_update(self, *args, **kwrds):
        raise NotImplementedError

### Phase 2: The SGD Optimizer (Child) 🏎️

# This class only handles the specific "physics" of the movement.
# 
# 1. **Inherit**: Extend the `BaseOptimizer`.
# 2. **Compute Update**:
# * Receive the `grad`, the `state` (the specific velocity buffer for this weight), and the `current_lr`.
# * **Update Velocity**: Calculate the new velocity: .
# * **Store**: Update the `state` array in-place so the change persists in the parent's dictionary.
# * **Return**: Send back the scaled step: .


class SGD(Optimizer):
    def __init__(self, lr=0.01, l1=0.0, l2=0.0, beta=0.9, scheduler=None):
        # Initialize the parent class first
        super().__init__(lr, l1, l2, scheduler)
        self.beta = beta

    def _compute_update(self, grad, state, current_lr):
        # 1. Update the state (velocity) in-place
        # Note: 'state' is a reference to the array in self.state[key]
        state[:] = self.beta * state + grad
        
        # 2. Return the step to be subtracted
        return current_lr * state
    
class RMSProp(Optimizer):
    def __init__(self, lr=0.01, rho=0.99, epsilon=1e-7, l1=0.0, l2=0.0, scheduler=None):
        super().__init__(lr, l1, l2, scheduler)
        self.rho = rho
        self.epsilon = epsilon

    def _compute_update(self, grad, state, current_lr):
        # Update the moving average of squared gradients
        state[:] = self.rho * state + (1 - self.rho) * np.square(grad)
        
        # Calculate the adaptive step
        return (current_lr * grad) / (np.sqrt(state) + self.epsilon)
    

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, l1=0.0, l2=0.0, scheduler=None):
        super().__init__(lr, l1, l2, scheduler)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _compute_update(self, grad, state, current_lr):
        # 1. Update first and second moments
        state["m"][:] = self.beta1 * state["m"] + (1 - self.beta1) * grad
        state["v"][:] = self.beta2 * state["v"] + (1 - self.beta2) * np.square(grad)

        # 2. Bias correction
        m_hat = state["m"] / (1 - self.beta1 ** self.iterations)
        v_hat = state["v"] / (1 - self.beta2 ** self.iterations)
        
        # 3. Final update calculation
        return (current_lr * m_hat) / (np.sqrt(v_hat) + self.epsilon)
    
    def _init_state(self, param):
        """Override the parent method because we need poth momentum and velocity"""
        return {
            "m": np.zeros_like(param), 
            "v": np.zeros_like(param)
        }

class AdamW(Adam):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, l1=0.0, l2=0.01, scheduler=None, weigh_decay = 0.01):
        # We call the Adam init but explicitly pass weight_decay=True
        super().__init__(
            lr=lr, 
            beta1=beta1, 
            beta2=beta2, 
            epsilon=epsilon, 
            l1=l1, 
            l2=l2, 
            scheduler=scheduler
        )
        self.weight_decay = weigh_decay