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

import gc
def print_tensor_sizes():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def same_storage(x, y):
    """
    x = torch.arange(10)
    y = x[1::2]
    print(same_storage(x, y)) # prints True
    z = y.clone()
    print(same_storage(x, z)) # prints False
    print(same_storage(y, z)) # prints False
    """
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)
