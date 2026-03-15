from .registry import DATASET_REGISTRY
from .base import DataModule

def build_dataset(cfg) -> DataModule:
    """
    Factory entrypoint. Now returns a structured DataModule object.
    """
    name = cfg.name.lower()

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' is not registered. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    # 1. Instantiate the specific loader (e.g., MNISTLoader)
    loader = DATASET_REGISTRY[name](cfg)
    
    # 2. get_data() handles: Cache Check -> (Download/Process/Save) -> Load
    return loader.get_data()