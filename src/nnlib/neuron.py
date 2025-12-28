"""Single Neuron implementation."""

import numpy as np
from activations import *
from base import Module
from loss import BCE

get_activation = {
    "relu": lambda: ReLU(),
    "sigmoid": lambda: Sigmoid(),
    "tanh": lambda: Tanh(),
    "linear": lambda: Linear(),
}


class Neuron(Module):
    """A single neuron in a neural network.
    the neuron should be able to :
        1. weights initialization.
        2. compute forward pass:
            a. save the inputs while training
            b. compute the z.
            c. compute the activation.
            d. return the activation.
    """

    def __init__(self, input_size: int, activation: str = "sigmoid") -> None:
        super().__init__()
        self.input_size = input_size
        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias = 0
        self.activation = get_activation[activation.lower()]()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # save the input for backward calculation
        self.cache = x                      # (m, n)
        # calculate the linear part of the neuron
        z = x @ self.weights + self.bias    # (m, 1) -> (m, n) @ (n, 1) + (1,)
        # calculate the non-linear part of the neuron
        a = self.activation.forward(z)      # (m, 1)
        return a

    def backward(self, grad_output: np.ndarray) -> np.ndarray: # (m, 1)
        da_dz = self.activation.backward(grad_output = grad_output) # (m, 1)
        dL_dz = da_dz # (m, 1)
        dL_dw = self.cache.T @ dL_dz    # (n, 1) -> (n, m) @ (m, 1)
        dL_db = np.sum(dL_dz, axis=0, keepdims=True)    # (1,) -> sum(m)
        dL_dx = dL_dz @ self.weights.T  # (m, n) -> (m, 1) @ (1, n)

        return dL_dw, dL_db, dL_dx
    
    def train(self,x:np.ndarray, y: np.ndarray, loss_fn, lr: float = 1e-2, epochs: int=100) -> None:
        """Train the neuron using gradient descent."""
        history = []
        for epoch in range(epochs):
            #1. forward pass
            y_pred = self.forward(x)

            # 2. compute the loss
            loss = loss_fn(y, y_pred)
            history.append(loss)
            
            # 3. backward pass (loss -> activation -> weights)
            grad_loss = loss_fn.backward(y, y_pred)
            dL_dw, dL_db, _ = self.backward(grad_loss)

            # 4. update weights and bias
            self.weights -= lr * dL_dw
            self.bias -= lr * dL_db

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

        self.history = history
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Retrun prediction without caching."""
        return self.activation.forward(x @ self.weights + self.bias)
if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification


    # 1. Generate Synthetic Data
    # 100 samples, 2 features (e.g., Age and Income), 2 classes (e.g., Buy vs Not Buy)
    X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.1,
    class_sep=1.5,
    random_state=42
    )
    # X = StandardScaler().fit_transform(X)  # Feature scaling
    #X = X / (np.max(X, axis=0) - np.min(X, axis=0))  # Min-Max Scaling
    y = y.reshape(-1, 1) # Ensure shape is (100, 1)

    # 2. Setup your Custom Neuron
    # Assuming Neuron class and BCE loss are imported
    input_size = X.shape[1]
    my_neuron = Neuron(input_size=input_size, activation="sigmoid")
    loss_fn = BCE(from_logits=False)

    # 3. Train Custom Neuron
    print("--- Training Custom Neuron ---")
    history = my_neuron.train(X, y, loss_fn, lr=0.1, epochs=2000)

    # 4. Train Scikit-Learn Logistic Regression
    print("\n--- Training Scikit-Learn ---")
    sk_model = LogisticRegression(C=np.inf) # No regularization to match our simple neuron
    sk_model.fit(X, y.ravel())

    # 5. Final Comparison
    print("\n--- Final Weights & Bias ---")
    print(f"Custom Weights: {my_neuron.weights.flatten()}")
    print(f"Sklearn Weights: {sk_model.coef_.flatten()}")
    print(f"Custom Bias:    {my_neuron.bias.flatten()}")
    print(f"Sklearn Bias:    {sk_model.intercept_}")

    # 6. Accuracy Check
    custom_preds = (my_neuron.predict(X) > 0.5).astype(int)
    sk_preds = sk_model.predict(X).reshape(-1, 1)

    custom_acc = np.mean(custom_preds == y)
    sk_acc = np.mean(sk_preds == y)

    print(f"\nCustom Accuracy: {custom_acc * 100:.2f}%")
    print(f"Sklearn Accuracy: {sk_acc * 100:.2f}%")