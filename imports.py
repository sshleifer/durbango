#IMPORTS at top of notebook
%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

ISO = "ISO-8859-1"

import os
import pickle
from tqdm import tqdm, tqdm_notebook, tnrange
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import *
import itertools
from collections import *
import funcy
from scipy.spatial.distance import cosine as cosine_distance
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.model_selection import StratifiedKFold, KFold, ParameterGrid
from sklearn.utils import shuffle
from numpy.testing import assert_array_equal
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
import re
from glob import glob
import pickle
import socket
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from durbango.torch_utils import  *
except ImportError:
    pass

from IPython.lib.display import FileLink
from durbango import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HOSTNAME = socket.gethostname()
mock_arr = np.array(range(25)).reshape(5,5)
mock_df = pd.DataFrame(np.ones((5,5)), columns=['a', 'b', 'c', 'd', 'e'])
mock_df['color'] = ['red', 'blue', 'red', 'blue', 'green']
mock_df['bool_col'] = [True, False, True, False, True]
np.set_printoptions(precision=5, linewidth=110, suppress=True)

from ipykernel.kernelapp import IPKernelApp
from IPython.display import Markdown, display, HTML
from IPython.core.interactiveshell import InteractiveShell
def in_notebook(): return IPKernelApp.initialized()
def printmd(string): display(Markdown(string))
import warnings

display(HTML("<style>.container { width:90% !important; }</style>"))

# pretty print only the last output of the cell
# InteractiveShell.ast_node_interactivity = "last_expr" # "all" for all
try:
    with warnings.catch_warnings():
        import eli5
        from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
        import seaborn as sns
        from PIL import Image
except ImportError:
    pass

Path.ls =  lambda self: sorted(list(self.iterdir()))
ParameterGrid.l = property(lambda self: list(self))
pd.Series.flipped_dict = property(lambda ser: funcy.flip(ser.to_dict()))

pd.DataFrame.dsort = descending_sort
pd.Series.dsort = descending_sort
pd.DataFrame.asort = ascending_sort
pd.Series.asort = ascending_sort

%alias_magic h history -p "-l 20 -u -g"
