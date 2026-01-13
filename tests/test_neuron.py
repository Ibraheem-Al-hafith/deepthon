from deepthon.nn.neuron import Neuron
from deepthon.nn.losses import BCE
import numpy as np
from typing import Any, List

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

NDArray = np.ndarray[Any]
def test_neuron():

    # 1. Generate Synthetic Binary Classification Data
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
    
    # Reshape targets to (Samples, 1) for matrix operations
    y_reshaped: NDArray = y.reshape(-1, 1).astype(np.float64)
    X_float: NDArray = X.astype(np.float64)

    # 2. Setup Custom Neuron and Loss
    input_dim: int = X_float.shape[1]
    my_neuron: Neuron = Neuron(input_size=input_dim, activation="sigmoid")
    loss_function: BCE = BCE(from_logits=False)

    # 3. Train Custom Neuron
    print("--- Training Custom Neuron ---")
    history_log: List[float] = my_neuron.fit(X_float, y_reshaped, loss_function, lr=0.1, epochs=2000)

    # 4. Train Scikit-Learn Logistic Regression for benchmarking
    print("\n--- Training Scikit-Learn LogisticRegression ---")
    sk_model: LogisticRegression = LogisticRegression(C=np.inf)
    sk_model.fit(X_float, y)

    # 5. Result Comparison
    print("\n--- Parameter Comparison ---")
    print(f"Custom Weights: {my_neuron.weights.flatten()}")
    print(f"Sklearn Weights: {sk_model.coef_.flatten()}")
    print(f"Custom Bias:    {my_neuron.bias.flatten()}")
    print(f"Sklearn Bias:    {sk_model.intercept_}")

    # 6. Accuracy Evaluation
    custom_preds: NDArray = (my_neuron.predict(X_float) > 0.5).astype(int)
    sk_preds: NDArray = sk_model.predict(X_float).reshape(-1, 1)

    custom_acc: float = float(np.mean(custom_preds == y_reshaped))
    sk_acc: float = float(np.mean(sk_preds == y_reshaped))

    print(f"\nCustom Accuracy: {custom_acc * 100:.2f}%")
    print(f"Sklearn Accuracy: {sk_acc * 100:.2f}%")

if __name__ == "__main__":
    test_neuron()