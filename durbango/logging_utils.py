import time

#from durbango.torch_utils import bytes_to_human_readable
from py3nvml import py3nvml
import torch
import psutil
import pandas as pd
import time
import os


def bytes_to_human_readable(memory_amount):
    """ Utility to convert a number of bytes (int) in a human readable string (with units)
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if memory_amount > -1024.0 and memory_amount < 1024.0:
            return "{:.3f}{}".format(memory_amount, unit)
        memory_amount /= 1024.0
    return "{:.3f}TB".format(memory_amount)


def run_gpu_mem_counter():
    # Sum used memory for all GPUs
    if not torch.cuda.is_available(): return 0
    py3nvml.nvmlInit()
    devices = list(range(py3nvml.nvmlDeviceGetCount())) #if gpus_to_trace is None else gpus_to_trace
    gpu_mem = 0
    for i in devices:
        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem += meminfo.used
    py3nvml.nvmlShutdown()
    return gpu_mem



class LoggingMixin:

    def log_mem(self, msg='', verbose=False):
        if not hasattr(self, 'logs'):
            self.reset_logs()
        self.logs.append(self.collect_log_data(msg=msg, verbose=verbose))

    def reset_logs(self):
        def resetter(module):
            module.logs = []
            module.t_init = time.time()
        self.apply(resetter)

    @property
    def log_df(self):
        if not hasattr(self, 'logs'):
            self.reset_logs()
        log_df = pd.DataFrame(self.logs)
        return log_df

    def save_log_csv(self, path):
        self.combine_logs().to_csv(path)

    @staticmethod
    def collect_log_data(msg='', verbose=False):
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss
        gpu_mem = run_gpu_mem_counter()
        record = dict(cpu_mem=cpu_mem, gpu_mem=gpu_mem,
                      time=time.time(),
                      msg=msg)
        long_msg = f'{msg}: GPU: {bytes_to_human_readable(gpu_mem)} CPU: {bytes_to_human_readable(cpu_mem)}'
        record['long_msg'] = long_msg
        if verbose:
            print(long_msg)
        return record
    def save_logs(self, path):
        strang = '\n'.join(self.combine_logs().long_msg.values)
        with open(path, 'w') as f:
            f.write(strang)

    def combine_logs(self):
        LOGS = [self.log_df]
        def get_child_logs(module):
            df = getattr(module, 'log_df', pd.DataFrame())
            LOGS.append(df)

        self.apply(get_child_logs)
        log_df =  pd.concat(LOGS).sort_values('time')

        return log_df.pipe(assign_diffs).sort_values('time')

    @property
    def summary(self):
        log_df = self.combine_logs()
        ranges = {x: log_df[x].max() - log_df[x].min() for x in ['cpu_mem', 'gpu_mem', 'time']}
        ranges['cpu_mem'] = bytes_to_human_readable(ranges['cpu_mem'])
        ranges['gpu_mem_chg'] = bytes_to_human_readable(ranges['gpu_mem'])
        ranges['gpu_mem_peak'] = bytes_to_human_readable(log_df['gpu_mem'].max())
        ranges['time'] = round(ranges['time'], 2)
        return pd.Series(ranges)

def assign_diffs(log_df):
    log_df['cpu_mem_delta'] = log_df['cpu_mem'].diff()
    log_df['gpu_mem_delta'] = log_df['gpu_mem'].diff()
    log_df['time_delta'] = log_df['time'].diff()
    return log_df


class LoggingModule(torch.nn.Module, LoggingMixin):  # can replace nn.Module inheritance!
    pass
