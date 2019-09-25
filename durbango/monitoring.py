import subprocess
import pandas as pd
import re

mem_col = 'memory_usage_(Gb)'
IGNORE_COLS = ['RSS', 'TTY', 'STAT', 'VSZ']

def get_active_processes_df(filter_command_pattern=None, ignore_cols=True) -> pd.DataFrame:
    '''return `ps aux` output command output as DataFrame
    Args:
        filter_command_pattern (str) or None: filter to only commands that contain this string
        ignore_cols: ignore ['RSS', 'TTY', 'STAT'] columns
    Returns: pd.DataFrame
    '''
    table = subprocess.check_output(['ps', 'aux', '-w', '-w']).decode('ascii')
    table = [re.split(r' +', r) for r in table.split('\n')]
    table = [r[:10] + [' '.join(r[10:])] for r in table[:-1]]
    df = pd.DataFrame(table[1:], columns=table[0])
    # Make datat Prettier
    df[mem_col] = (df['VSZ'].astype(float) / 1e9).round(1)

    if filter_command_pattern is not None:
        df = df[df['COMMAND'].str.contains(filter_command_pattern)]
    if IGNORE_COLS:
        df = df.drop(ignore_cols, errors='ignore', axis=1)
    return df.reset_index(drop=True).sort_values(mem_col, ascending=False)
