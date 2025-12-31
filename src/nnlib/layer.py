"""Initialization for the nn weights"""
from activations import *
import numpy as np
from typing import Literal, Tuple, List
from base import Module
from optimization import Dropout
from loss import BCE
from sklearn.datasets import make_circles
from optimizers import SGD


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


# =====================================================================
#************************   MLP   *******************************
# =====================================================================

class Sequential(Module):
    """
    Multi Layer Perceptron
    """
    def __init__(self, layers: List[Layer | Dropout | Activation], optimizer):
        super().__init__()
        self.layers = layers
        self.optimizer = optimizer
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

    def update(self):
        self.optimizer.step(self.layers)
    def add(self, layers: Layer | Dropout | Activation | List[Layer | Dropout | Activation]):
        if isinstance(layers, list):
            for layer in layers:
                self.add(layer)
        else:
            self.layers.append(layers)


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

def train_model(epochs=1000, lr=0.3, n_samples=1000):
    # Generate dataset
    X, y = make_circles(n_samples=n_samples, noise=0.1, random_state=42)
    y = y.reshape(-1, 1).astype(np.float32)
    
    optimizer = SGD(lr = lr, l1 = 0, l2=0.001)
    # Create model
    model = Sequential([], optimizer=optimizer)
    model.add([
        Layer(2, 64, activation="relu"),
        Dropout(0.1),
    ])
    model.add(Layer(64, 1, "sigmoid"))
    
    # Loss and optimizer
    loss_fn = BCE(from_logits=False)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        output = model(X)
        if np.any(np.isnan(output)):
            print("NaN in output")
            break
        loss = loss_fn(output, y)
        
        # Backward pass
        grad = loss_fn.backward(y, output)
        model.backward(grad)
        
        # Update weights
        model.update()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Evaluate
    model.eval()
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        predictions = (model(X) > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Final Accuracy: {accuracy:.4f}")
    
    return model

def test_layer():
    from sklearn.datasets import make_circles
    X, y = make_circles(5)
    model = Sequential(
        [Layer(2, 32),
        ReLU(),
        Layer(32, 32,"relu"),
        Dropout(0.1),
        Layer(32, 1, "sigmoid")], optimizer=SGD()
    )
    model.eval()
    output = model(X)
    #grad = model.backward(np.array([0.8]))
    print(
        f"output : {output}",
    )

if __name__ == "__main__":
    # Run the comparison
    # compare_with_pytorch()
    # test_layer()
    train_model(1000,0.3)