
# 🧠 **deepthon & deepthon_pipeline**

### *A Minimal Deep Learning Framework & End-to-End Pipeline Built from Scratch with NumPy*

<div align="center">
<img src="assets/img.png" alt="Deepthon Pipeline Header" width="600">

## **Research-oriented • Transparent • Modular • Lightweight**
</div>


## 📜 Abstract

**deepthon** is a **from-scratch neural network framework** designed to expose the mathematical systems behind modern AI. While production frameworks hide details behind automatic differentiation, **deepthon** treats neural networks as transparent numerical systems.

**deepthon_pipeline** is the production-ready automation layer built on top of this engine. It demonstrates a complete machine learning lifecycle—from raw data ingestion to interactive deployment—using a clean, hackable design.

---

## 🧩 Key Features & Benchmarks

* **Mathematical Transparency:** Implements forward propagation, backpropagation, and gradient-based optimization using only NumPy.
* **Modular Pipeline:** A standardized workflow for data cleaning, stratified binning, and evaluation.
* **Built-in Benchmarks:**
* 🩺 **Breast Cancer:** Binary classification for medical diagnosis.
* ✍️ **MNIST:** Multi-class handwritten digit recognition.
* ⚡ **Turbine Energy:** Regression for industrial sensor data.



| Feature | deepthon Ecosystem | Standard Frameworks |
| --- | --- | --- |
| **Autograd** | ❌ Under Dev | ✅ Yes |
| **Transparency** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Debug Mode** | Mathematics-first | Op-code-first |
| **Dependencies** | NumPy only | Heavy |

---

## 🛠️ Installation & Setup

We recommend using **uv** for fast, reproducible environment management.

### 🐧 Linux / 🍎 Mac

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone https://github.com/Ibraheem-Al-hafith/deepthon_pipeline.git
cd deepthon_pipeline
uv sync

```

### 🪟 Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
git clone https://github.com/Ibraheem-Al-hafith/deepthon_pipeline.git
cd deepthon_pipeline
uv sync

```

---

## 🖥️ Usage Modes

### 1. Interactive GUI

Launch the **Gradio** dashboard to test models in real-time with a visual interface.

```bash
python -m src.deepthon_pipeline.ui.app serve

```

### 2. CLI Training & Testing

Run experiments using modular YAML configurations.

```bash
# Train using a config file
python -m src.deepthon_pipeline.cli.main train configs/config.yaml

# Evaluate specific checkpoints
python -m src.deepthon_pipeline.cli.main test configs/config.yaml runs/exp/model.npz cancer tiny

```

---

## ⚙️ Example Configuration (`config.yaml`)

```yaml
experiment: research_v1
datasets:
  cancer:
    name: cancer
    input_dim: 30
    output_dim: 1
    train_config:
      loss_fn: BCE
      metric: f1

model:
  tiny:
    type: sequential
    architecture:
      - [null, 64, relu]
      - [batchnorm, 64]
      - [Dropout, 0.2]
      - [64, null, linear]

```

---

## 🗂️ Project Structure

```text
deepthon_pipeline/
├─ configs/      📄 Experiment YAML templates
├─ src/
│  └─ deepthon/  🧠 Core Framework (Layers, Optimizers, Losses)
│  └─ pipeline/  📦 Data Loaders, Runner & Trainer logic
│  └─ cli/       🖥️ Command Line Interface
│  └─ ui/        📺 Gradio Web Interface
└─ tests/        🧪 Pytest suite

```

---

## 🛣️ Roadmap

* [ ] Fully integrated Autograd (Automatic Differentiation)
* [ ] Convolutional Neural Network (CNN) support
* [ ] Training visualization dashboard with MLflow
* [ ] GPU acceleration via CuPy

---

## 📜 License & Acknowledgment

Distributed under the **MIT License**. This project was developed as part of a commitment to making deep learning research accessible and mathematically transparent.

**Happy hacking! 🚀🧠**

---
