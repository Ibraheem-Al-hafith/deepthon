from pathlib import Path
from src.deepthon_pipeline.cli.commands import load_config
from src.deepthon_pipeline.data.data.base import *
def test_mnist_loading(config_path):
    cfg = load_config(config_path)
    loader = MNISTLoader(cfg.datasets.mnist)
    loader.load()

def test_cancer_loading(config_path):
    cfg = load_config(config_path)
    loader = BREAST_CANCER_LOADER(cfg.datasets.__getattr__("cancer"))
    loader.load()
def test_turbines_loading(config_path):
    cfg = load_config(config_path)
    loader = TURBINES_LOADER(cfg.datasets.turbines)
    loader.load()
cfg = "configs/mnist.yaml"