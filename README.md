# deepthon

**deepthon** is a lightweight deep-learning library implemented from scratch using **NumPy**.  
It provides modular neural-network building blocks, training utilities, and optimization tools designed for learning, experimentation, and research.

---

## Features

- Sequential model API with composable layers
- Forward / backward propagation implemented manually
- Activations, losses, optimizers, and LR schedulers
- Simple training pipeline with metrics and validation
- Minimal dependency footprint (NumPy only)

---
## Prerequisites :
- NumPy:
```bash
pip install numpy
```

## Installation

From source (working):

```bash
git clone https://github.com/Ibraheem-Al-hafith/deepthon
cd deepthon
pip install -e .
```

From PyPI (when published):

```bash
pip install deepthon
````

---

## Quick Start

```python
import numpy as np
from deepthon.nn import Sequential, Layer
from deepthon.nn.activations import ReLU, Sigmoid
from deepthon.nn.losses import BCE
from deepthon.nn.optimizers import Adam
from deepthon.pipeline import Trainer
from deepthon.utils.metrics import Accuracy

X = np.random.randn(500, 2)
y = (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)
    
model = Sequential([
    Layer(2, 16, activation=ReLU()),
    Layer(16, 8, activation=ReLU()),
    Layer(8, 1, activation=Sigmoid()),
])
    
optimizer = Adam(lr=1e-3)
loss = BCE()
    
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_func=loss,
    batch_size=32,
    metric_fn="accuracy",
)
    
trainer.train(X, y, epochs=20)
```

More examples are available in `examples/`.

---

## Public API

### Core Modules

| Module              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `deepthon.nn`       | Layers, models, activations, losses, optimizers, schedulers |
| `deepthon.pipeline` | Training utilities and Trainer class                        |
| `deepthon.utils`    | Metrics and dataset helpers                                 |

Typical imports:

```python
from deepthon.nn import Sequential, Layer
from deepthon.nn import activations, losses, optimizers, schedulers
from deepthon.pipeline import Trainer
from deepthon.utils import metrics
```

---

## Project Structure

```text
src/deepthon
├─ nn/          Core neural-network components
├─ pipeline/    Training pipeline
├─ utils/       Metrics and helpers
```

---

## Contributing

Contributions are welcome.
Please follow the coding style, add tests for new features, and open an issue before large changes.

---

## License

MIT License — see `LICENSE` for details.

---

## Links

* Repository: [deepthon](https://github.com/Ibraheem-Al-hafith/deepthon)
* Issues: [issues](https://github.com/Ibraheem-Al-hafith/deepthon/issues)

