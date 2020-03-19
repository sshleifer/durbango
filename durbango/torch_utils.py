import torch
DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from durbango.nb_utils import is_iterable

def num_parameters(module, only_trainable: bool = False) -> int:
    """
    Get number of (optionally, trainable) parameters in the module.
    """
    params = filter(lambda x: x.requires_grad, module.parameters()) if only_trainable else module.parameters()
    return sum(p.numel() for p in params)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def get_shapes(x):
    """Recursive"""
    if hasattr(x, 'shape'):
        return x.shape
    elif isinstance(x, dict):
        return {k: get_shapes(v) for k,v in x.items()}
    elif is_iterable(x):
        return [get_shapes(v) for v in x]
    else:
        return None
