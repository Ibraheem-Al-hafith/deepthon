from deepthon.nn.optimizers import SGD, Adam, AdamW

def build_optimizer(ocfg):

    if ocfg.name.lower() == "sgd":
        return SGD(lr=ocfg["lr"])

    if ocfg.name.lower() == "adam":
        return Adam(lr=ocfg.lr)
    
    if ocfg.name.lower() == "adamw":
        return AdamW(lr=ocfg.lr)

    raise ValueError("Unsupported optimizer")
