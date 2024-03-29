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

def print_shape(obj):
    if torch.is_tensor(obj): return obj.shape
    elif isinstance(obj, (list, tuple)): return [print_shape(x) for x in obj]
    elif isinstance(obj, dict): return {k: print_shape(v) for k, v in obj.items()}
    else: return obj


def avg_checkpoints(sds):
    new_sd = {}
    for k in sds[0].keys():
        new_sd[k] = torch.mean([sd[k] for sd in sds])
    return new_sd


def get_shapes(x):
    """Recursive"""
    if hasattr(x, 'shape'):
        return tuple(x.shape)
    elif isinstance(x, dict):
        return {k: get_shapes(v) for k,v in x.items()}
    elif is_iterable(x):
        return [get_shapes(v) for v in x]
    else:
        return None

def get_tensor_shapes_and_pointers(x):
    """Recursive"""
    if isinstance(x, torch.Tensor):
        return (x.shape, x.data_ptr())
    elif isinstance(x, dict):
        return {k: get_shapes(v) for k,v in x.items()}
    elif is_iterable(x):
        return [get_shapes(v) for v in x]
    else:
        return None

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def local_sizeof():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

import gc, inspect

def find_names(obj):
    frame = inspect.currentframe()
    for frame in iter(lambda: frame.f_back, None):
        frame.f_locals
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names


import gc

from .nb_utils import tqdm_nice
from collections import defaultdict
import pandas as pd
def print_tensor_sizes(ignore_names = ('obj', 'weight', 'bias')):
    results = []
    seen_ptrs = set()
    for obj in tqdm_nice(gc.get_objects()):
        try:
            assert isinstance(obj, torch.Tensor)
            ptr = obj.data_ptr()
            if ptr in seen_ptrs: continue
            seen_ptrs.add(ptr)
            names = [x for x in find_names(obj) if x not in ignore_names]
            for name in names:
                results.append((name, obj.numel(), ptr,  obj.dtype, obj.device))
        except AssertionError:
            pass
    colnames = ['varname', 'numel', 'data_ptr', 'data_type', 'device']
    return pd.DataFrame(results, columns=colnames).sort_values('numel', ascending=False)

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


def compare_state_dict(dct_a, dct_b):
    SENTINEL = torch.zeros(3)
    k1, k2 = set(dct_a), set(dct_b) # just the keys
    deltas = []
    for k in tqdm_nice(k1.union(k2)):
        vala, valb = dct_a.get(k, SENTINEL), dct_b.get(k, SENTINEL)
        if vala.shape == valb.shape and torch.eq(vala, valb).all():
            continue
        else:
            deltas.append((k, vala.numel(), valb.numel()))
    return pd.DataFrame(deltas, columns=['key', 'numel_a', 'numel_b'])

def log_tensor(msg, x):
    sq = x.squeeze()
    if sq.ndim == 2:
        slice = x[:3, :4]
    elif sq.ndim == 3:
        slice = x[:, 0, :6]
    else:
        slice = x[:5]
    print(f"{msg}: shape: {x.shape} min: {x.min(): .4f} max: {x.max(): .4f} slice: {slice}")

from .nb_utils import remove_prefix
def convert_pl_to_hf(pl_ckpt_path, hf_model, save_path):
    state_dict = {remove_prefix(k, 'model.'): v for k, v in
           torch.load(pl_ckpt_path, map_location='cpu')['state_dict'].items()}
    missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
    assert not missing, f'missing keys: {missing}'
    hf_model.save_pretrained(save_path)


def get_src_lens(tok, examples):
    src_lens = tok(examples, padding='longest', truncation=True, return_tensors='pt').input_ids.ne(tok.pad_token_id).sum(1)
    return src_lens


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def count_parameters(model):
    model_parameters = model.parameters()
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
