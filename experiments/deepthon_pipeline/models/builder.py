from deepthon.nn.layers import Sequential, Layer, Dropout, BatchNorm
from .registry import register_model

# ---------------- Block mapper registery -----------------
BLOCK_MAPPER = {}
def register_block(name):
    def wrapper(fn):
        BLOCK_MAPPER[name.lower()] = fn
        return fn
    return wrapper

# ------- Block Implementation -------------

@register_block("layer")
def map_linear(token):
    """
    Formats:
    [in_dim, out_dim]
    [in_dim, out_dim, activation]
    """
    in_dim, out_dim, *rest = token
    activation = rest[0] if rest else None
    return [Layer(in_dim, out_dim, activation=activation)]

@register_block("dropout")
def map_dropout(token):
    """
    ["dropout",p]
    """
    p = token[1] if len(token)>0 else 0.2
    return [Dropout(p)]

@register_block("batchnorm")
def map_batchnorm(token):
    """
    ["batchnorm", num_features]
    """
    num_features = token[1]
    return [BatchNorm(num_features=num_features)]

# ------ Dispatcher ----------
def map_block(token, input_dim: int | None = None, output_dim: int | None = None):
    """Route token to correct mapper and resolve dimensions dynamically."""
    
    # 1. Handle non-layer blocks (dropout, batchnorm)
    if isinstance(token[0], str):
        key = token[0].lower()
        if key not in BLOCK_MAPPER:
            raise ValueError(f"Unknown architecture block: {token}")
        return BLOCK_MAPPER[key](token)
    
    # 2. Handle Layer blocks: [in, out, act]
    # We work on a copy to avoid mutating the global config!
    resolved_token = list(token)
    
    if resolved_token[0] is None: 
        resolved_token[0] = input_dim
    if resolved_token[1] is None: 
        resolved_token[1] = output_dim
        
    return BLOCK_MAPPER['layer'](resolved_token)

# -------- build sequential model --------

def build_sequential_from_arch(arch, input_dim: int | None = None, output_dim: int | None = None):
    layers = []
    for block in arch:
        layers.extend(map_block(block, input_dim=input_dim, output_dim=output_dim))
    return Sequential(layers)

# ---------- Public Model Builder ----------

@register_model("sequential")
def build_model_from_config(cfg, input_dim: int | None = None, output_dim: int | None = None, model_name: str | None = None):
    """
    Expected YAML:

    model:
      name: sequential
      architecture:
        - [null, 64, "relu"]
        - ["batchnorm"]
        - [64, 32, "relu"]
        - ["dropout", 0.5]
        - [32, null]
    """
    # Fetch architecture reference
    model_cfg = cfg.model.__getattr__(model_name) if model_name else cfg.model
    arch = model_cfg.architecture
    
    return build_sequential_from_arch(arch, input_dim=input_dim, output_dim=output_dim)
        