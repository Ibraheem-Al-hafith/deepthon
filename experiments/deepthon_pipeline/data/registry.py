DATASET_REGISTRY = {}

def register_dataset(name):
    """
    Decorator to register a dataset builder.
    Usage:
        @register_dataset("mnist")
        def build_mnist(cfg): ...
    """
    def wrapper(fn):
        DATASET_REGISTRY[name.lower()] = fn
        return fn
    return wrapper