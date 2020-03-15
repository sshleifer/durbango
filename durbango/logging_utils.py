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

    def log_mem(self, msg='', verbose=True):
        if not hasattr(self, 'logs'):
            self.reset_logs()
        self.logs.append(self.collect_log_data(msg=msg, verbose=verbose))

    def reset_logs(self):
        self.logs = []
        self.t_init = time.time()

    @property
    def log_df(self):
        if not hasattr(self, 'logs'):
            self.reset_logs()
        log_df = pd.DataFrame(self.logs)
        log_df['time_since_init'] = log_df['time'] - self.t_init
        return log_df

    def save_log_csv(self, path):
        self.log_df.to_csv(path)

    @staticmethod
    def collect_log_data(msg='', verbose=False):
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss
        gpu_mem = run_gpu_mem_counter()
        record = dict(cpu_mem=cpu_mem, gpu_mem=gpu_mem,
                      time=time.time(),
                      msg=msg)
        long_msg = f'{msg}: GPU: {bytes_to_human_readable(gpu_mem)} CPU: {bytes_to_human_readable(gpu_mem)}'
        record['long_msg'] = long_msg
        if verbose:
            print(long_msg)
        return record

    def combine_logs(self):
        LOGS = [self.log_df]
        def get_child_logs(module):
            df = getattr(module, 'log_df', pd.DataFrame())
            LOGS.append(df)

        self.apply(get_child_logs)
        log_df =  pd.concat(LOGS).sort_values('time')

        return log_df.pipe(assign_diffs)

def assign_diffs(log_df):
    log_df['cpu_mem_delta'] = log_df['cpu_mem'].diff()
    log_df['gpu_mem_delta'] = log_df['gpu_mem'].diff()
    log_df['time_delta'] = log_df['time'].diff()
