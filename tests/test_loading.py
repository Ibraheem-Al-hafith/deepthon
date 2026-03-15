from pathlib import Path
from experiments.deepthon_pipeline.cli.commands import load_config
from experiments.deepthon_pipeline.data.base import MNISTLoader, BreastCancerLoader, TurbinesLoader
cfg = "experiments/configs/config.yaml"

def test_mnist_loading(config_path = cfg):
    cfg = load_config(config_path)
    loader = MNISTLoader(cfg.datasets.mnist)
    loader.get_data()

def test_cancer_loading(config_path = cfg):
    cfg = load_config(config_path)
    loader = BreastCancerLoader(cfg.datasets.__getattr__("cancer"))
    loader.get_data()
def test_turbines_loading(config_path = cfg):
    cfg = load_config(config_path)
    loader = TurbinesLoader(cfg.datasets.turbines)
    loader.get_data()