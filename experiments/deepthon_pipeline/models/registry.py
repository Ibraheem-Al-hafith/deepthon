MODEL_REGISTRY = {}


def register_model(name):
    def wrapper(fn):
        MODEL_REGISTRY[name.lower()] = fn
        return fn
    return wrapper


def build_model(cfg):
    model_cfg = cfg["model"]

    name = model_cfg.get("name")
    if name is None:
        raise ValueError("Model config must include `name`")

    name = name.lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {name}")

    return MODEL_REGISTRY[name](cfg)
