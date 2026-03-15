import yaml
from pathlib import Path

class ConfigNode(dict):
    """Dictionary -> object-style configuration tree."""
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            value = ConfigNode(value)
        return value
    
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delattr__

def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return ConfigNode(raw)
