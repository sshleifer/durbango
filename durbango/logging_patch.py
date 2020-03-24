import os
import time

import pandas as pd

from .logging_utils import bytes_to_human_readable, collect_log_data, assign_diffs


def log_mem(self, msg='', verbose=False):
    if not hasattr(self, 'logs'):
        self.reset_logs()
    self.logs.append(collect_log_data(msg=msg, verbose=verbose))


def reset_logs(self):
    def resetter(module):
        module.logs = []
        module.t_init = time.time()
    self.apply(resetter)


def save_logs(self, path):
    strang = '\n'.join(self.combine_logs().long_msg.values)
    with open(path, 'w') as f:
        f.write(strang)


def combine_logs(self):
    LOGS = [self.log_df]
    def get_child_logs(module):
        df = getattr(module, 'log_df', pd.DataFrame())
        LOGS.append(df)


def summary_fn(self):
    log_df = self.combine_logs()
    ranges = {x: log_df[x].max() - log_df[x].min() for x in ['cpu_mem', 'gpu_mem', 'time']}
    ranges['cpu_mem'] = bytes_to_human_readable(ranges['cpu_mem'])
    ranges['gpu_mem_chg'] = bytes_to_human_readable(ranges['gpu_mem'])
    ranges['gpu_mem_peak'] = bytes_to_human_readable(log_df['gpu_mem'].max())
    ranges['time'] = round(ranges['time'], 2)
    return pd.Series(ranges)

from types import MethodType
def patch_module_with_memory_mixin(model):
    """create logging methods using MethodType"""
    _method = lambda f: MethodType(f, model)
    model.reset_logs = _method(reset_logs)
    model.save_logs = _method(save_logs)
    model.save_log_csv = _method(save_log_csv)
    #model.log_mem = _method(log_mem)
    cls = type(model)
    cls.log_df = property(log_df_fn)
    cls.log_mem = _method(log_mem)
    model.summary = property(summary_fn)
    model.combine_logs = _method(combine_logs)


def log_df_fn(self):
    if not hasattr(self, 'logs'):
        self.reset_logs()
    log_df = pd.DataFrame(self.logs)
    return log_df




def save_log_csv(self, path):
    self.combine_logs().to_csv(path)


