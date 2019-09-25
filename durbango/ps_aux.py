import subprocess
import pandas as pd
import re

def get_ps_aux():
    '''return `ps aux` command as dataframe'''
    table = subprocess.check_output(['ps', 'aux', '-w', '-w']).decode('ascii')
    table = [re.split(r' +', r) for r in table.split('\n')]
    table = [r[:10] + [' '.join(r[10:])] for r in table[:-1]]
    return pd.DataFrame(table[1:], columns=table[0])


def get_active_kernels():
    '''return dataframe containing active ipython kernel processes'''
    df = get_ps_aux()
    df = df[df['COMMAND'].str.contains('python')]
#     df['kernel_id'] = df['COMMAND'].str.split('kernel-').str[1].str.rstrip('.json')
    df['memory_usage_(Gb)'] = (df['VSZ'].astype(float) / 1e6).round(1)
    df = df.sort_values(by='memory_usage_(Gb)', ascending=False)
    return df.drop(['RSS', 'TTY', 'STAT', 'VSZ'], 1).reset_index(drop=True)

df = get_active_kernels()
