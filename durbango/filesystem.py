import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

Path.ls = lambda self: sorted(list(self.iterdir()))
from .nb_utils import tqdm_nice


def read_pickle(path, lib=pickle):
    """pickle.load(path)"""
    with open(path, 'rb') as f:
        return lib.load(f)


def write_pickle(obj, path, lib=pickle):
    """pickle.dump(obj, path)"""
    with open(path, 'wb') as f:
        return lib.dump(obj, f)


def dill_load(path):
    import dill
    return read_pickle(path, lib=dill)


def dill_save(obj, path):
    import dill
    return write_pickle(obj, path, lib=dill)


def pickle_load_gzip(path):
    import gzip
    with gzip.open(path, 'rb') as f:
        return pickle.load(f, encoding='latin-1')


# add some aliases to alleviate confusion
pickle_read = read_pickle
pickle_load = read_pickle
save_pickle = write_pickle
pickle_save = write_pickle


def make_directory_if_not_there(path) -> None:
    Path(path).mkdir(exist_ok=True)


def mpath(path, parents=True, exist_ok=True) -> Path:
    """Make a directory at path, and return it."""
    p = Path(path)
    p.mkdir(exist_ok=exist_ok, parents=parents)
    return p


def tar_compress(folder_path, save_path=None):
    import tarfile
    folder_path = Path(folder_path)
    if save_path is None: save_path = f'{folder_path}.tgz'
    with tarfile.open(save_path, "w:gz") as tar:
        if folder_path.is_dir():
            for name in tqdm_nice(folder_path.ls):
                tar.add(str(name))
        else:
            tar.add(str(folder_path))


def get_git_rev(root='.'):
    '''Try to return the current git rev-parse HEAD, or None if cant find.'''
    import git
    try:
        git_ = git.Git(root)
        current_commit = git_.rev_parse('HEAD')
        return current_commit
    except Exception:
        print(f'Couldnt find git rev error: {e}')
        return None

def get_cur_branch(path='.'):
    import git
    repo = git.Git(path)
    cands = [x.lstrip('* ') for x in repo.branch().split('\n') if x.startswith('*')]
    assert len(cands) == 1
    return cands[0]
