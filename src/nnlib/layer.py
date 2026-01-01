"""Initialization for the nn weights"""
from .activations import *
import numpy as np
from typing import Literal, Tuple, List
from .base import Module


# =====================================================================
#************************   UTILITIES   *******************************
# =====================================================================
def HE(shape: Tuple[int, int]) -> np.ndarray:
    """HE initialization for ReLU and startups"""
    n_in, _ = shape[0], shape[1]
    scale = np.sqrt(2 / n_in)
    weights: np.ndarray = np.random.randn(*shape) * scale
    return weights

def Xavier(shape: Tuple[int, int], dist: Literal["normal", "uniform"] = "normal") -> np.ndarray:
    """Xavier initialization for sigmoid/tanh"""
    n_in, n_out = shape[0], shape[1]
    if dist == "normal":
        scale = np.sqrt(2 / (n_in + n_out))
        weights: np.ndarray = np.random.randn(*shape) * scale
    elif dist == "uniform":
        limit = np.sqrt(6 / (n_in+n_out))
        weights: np.ndarray = np.random.uniform(low= -limit, high=limit, size=shape)
    else:
        raise ValueError("The value of dist should be either 'normal' or 'uniform'")
    return weights

initialize = {
    "random": lambda shape: np.random.randn(*shape) * 0.01,
    "relu": HE,
    "sigmoid": Xavier,
    "tanh": Xavier,
}

get_activation = {
    "relu": lambda: ReLU(),
    "sigmoid": lambda: Sigmoid(),
    "tanh": lambda: Tanh(),
    "linear": lambda: Linear(),
}


# =====================================================================
#************************   Layer   *******************************
# =====================================================================

class Layer(Module):
    """
    Dense Layer
    """
    def __init__(
            self, n_inputs: int, n_neurons: int,
            activation: Literal["relu", "sigmoid", "tanh", "linear"] | Activation | None = None,
            ) -> None:
        """
        :param n_inputs: number of the input features
        :type n_inputs: int 
        :param n_neurons: number of the layer neurons
        :type n_neurons: int
        :param activation: activation function for the Layer
        :type activation: Literal["relu", "sigmoid", "tanh", "linear"] | None | Activation (must have forward and backward methods)
        """
        super().__init__()
        # 1. initialize the weights :
        self.weights: np.ndarray = initialize[
            str(activation) if activation in initialize.keys()
            else "sigmoid" if n_neurons == 1
            else "relu"
            ]((n_inputs, n_neurons)) # HE initialization as default, Xavier if one neuron, other wise based on the activation function
        self.bias = np.zeros((1, n_neurons))
        # 2. initialize the activation function :
        self.activation = get_activation[activation or "linear"]() if not isinstance(activation, Activation) else activation# get the activation function, default is linear (or no activation)
        self.dw = None
        self.db = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: data to be forwarded, the shape is (number of samples * number of features)
        :type x: np.ndarray
        """
        # 1. pass the x through the model to get the linear transformation (z)
        z:np.ndarray  = x @ self.weights + self.bias # (m,n) @ (n, nn) + (nn, 1) = (m, nn)
        # 2. pass the z through the non-linear part to get the activation (a)
        a:np.ndarray = self.activation(z)
        # 3. Handle cache saving the activation a for backward propagation in the case of training
        if self.training:
            self.x = x
        # 4. return the activations
        return a #(m, nn)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # 1. get the dL_dz from the activation backward
        dL_dz: np.ndarray = self.activation.backward(grad_output) # (m, nn)
        # 2. get the dw, db and dx from dz using chain rule
        self.dw = self.x.T @ dL_dz    #(n, m) @ (m, nn) = (n, nn)
        self.db = np.sum(dL_dz,axis=0,keepdims = True) # (m, nn) -> (1, nn)
        dL_dx:np.ndarray = dL_dz @ self.weights.T # (m, nn) @ (nn, n) -> (m, n)

        return dL_dx
    def get_parameters(self): 
        return [
            {"param": self.weights, "grad": self.dw, "name": "weight"},
            {"param": self.bias, "grad": self.db, "name": "bias"} 
        ]


# =============================================================================
# *************************** DROP OUT ****************************************
# =============================================================================

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
        self.keep_prob = 1 - self.drop_prob
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # If model mode is not training, return the original data
        if not self.training:
            return x
        # else, randomly drop a portion from the data
        # 1. calculate the mask
        mask: np.ndarray = (np.random.rand(*x.shape) < self.keep_prob).astype(np.float32)
        # 2. calculate the output and scale the output to serve the signal strength
        output:np.ndarray = (mask * x) / self.keep_prob
        # 3. store the mask for backward
        self.mask = mask

        return output
    def backward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            return x * self.mask / self.keep_prob
        else:
            return x


# =====================================================================
#************************   MLP   *******************************
# =====================================================================

class Sequential(Module):
    """
    Multi Layer Perceptron
    """
    def __init__(self, layers: List[Module]):
        super().__init__()
        self.layers = layers
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()
    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    def add(self, layers: Module| List[Module]):
        if isinstance(layers, list):
            for layer in layers:
                self.add(layer)
        else:
            self.layers.append(layers)

import numpy as np

class BatchNorm(Module):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Parameters for inference (Running averages)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Gradients
        self.dgamma = None
        self.dbeta = None
        
        # Mode
        self.training = True

    def forward(self, X):
        if self.training:
            # 1. Calculate Batch Statistics
            batch_mean = np.mean(X, axis=0, keepdims=True)
            batch_var = np.var(X, axis=0, keepdims=True)
            
            # 2. Update Running Statistics for Inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # 3. Normalize
            self.X_centered = X - batch_mean
            self.std_inv = 1.0 / np.sqrt(batch_var + self.epsilon)
            self.X_hat = self.X_centered * self.std_inv
        else:
            # Use Running Stats during Evaluation
            X_centered = X - self.running_mean
            std_inv = 1.0 / np.sqrt(self.running_var + self.epsilon)
            self.X_hat = X_centered * std_inv

        # 4. Scale and Shift
        return self.gamma * self.X_hat + self.beta

    def backward(self, dout):
        batch_size = dout.shape[0]
        
        # Gradient w.r.t. gamma and beta
        self.dgamma = np.sum(dout * self.X_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient w.r.t. input (X) - Simplified chain rule version
        dx_hat = dout * self.gamma
        da = (1.0 / batch_size) * self.std_inv * (
            batch_size * dx_hat - np.sum(dx_hat, axis=0, keepdims=True) -
            self.X_hat * np.sum(dx_hat * self.X_hat, axis=0, keepdims=True)
        )
        return da

    def get_parameters(self):
        return [
            {"param": self.gamma, "grad": self.dgamma, "name": "gamma"},
            {"param": self.beta, "grad": self.dbeta, "name": "beta"}
        ]



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

def compare_with_pytorch():
    # 1. Setup dimensions
    import numpy as np
    import torch
    import torch.nn as nn
    n_samples, n_inputs, n_neurons = 3, 4, 2
    
    # 2. Initialize your Layer (using Linear/None for direct comparison)
    # Note: Using your class logic from the previous snippet
    my_layer = Layer(n_inputs, n_neurons, activation="linear")
    my_layer.training = True # Enable caching for backward
    
    # 3. Initialize PyTorch Layer
    pt_layer = nn.Linear(n_inputs, n_neurons)
    
    # 4. SYNC WEIGHTS: PyTorch stores weights as (out, in), yours is (in, out)
    pt_layer.weight.data = torch.from_numpy(my_layer.weights.T).float()
    pt_layer.bias.data = torch.from_numpy(my_layer.bias).float()
    
    # 5. Prepare Input Data
    x_np = np.random.randn(n_samples, n_inputs).astype(np.float32)
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    
    # --- FORWARD PASS ---
    out_my = my_layer.forward(x_np)
    out_pt = pt_layer(x_pt)
    
    print("--- Forward Pass Check ---")
    print(f"Difference: {np.abs(out_my - out_pt.detach().numpy()).max():.2e}")
    
    # --- BACKWARD PASS ---
    # Create a dummy "upstream gradient" (dL/dY)
    grad_output_np = np.ones((n_samples, n_neurons), dtype=np.float32)
    grad_output_pt = torch.from_numpy(grad_output_np)
    
    # Your backward
    dL_dw_my, dL_db_my, dL_dx_my = my_layer.backward(grad_output_np)
    
    # PyTorch backward
    out_pt.backward(grad_output_pt)
    
    print("\n--- Backward Pass Check (Gradients) ---")
    # Compare Weight Gradients (Remember PyTorch is transposed)
    dw_diff = np.abs(dL_dw_my - pt_layer.weight.grad.numpy().T).max() # type: ignore
    print(f"Weight Grad Difference: {dw_diff:.2e}")
    
    # Compare Bias Gradients (PyTorch bias grad is (n_neurons,), yours is (1, n_neurons))
    db_diff = np.abs(dL_db_my.flatten() - pt_layer.bias.grad.numpy()).max() # type: ignore
    print(f"Bias Grad Difference:   {db_diff:.2e}")
    
    # Compare Input Gradients
    dx_diff = np.abs(dL_dx_my - x_pt.grad.numpy()).max() # type: ignore
    print(f"Input Grad Difference:  {dx_diff:.2e}")
if __name__ == "__main__":
    # Run the comparison
    compare_with_pytorch()
    test_dropout()
