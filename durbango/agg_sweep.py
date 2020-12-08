
from pathlib import Path
from glob import glob
import pandas as pd
from fire import Fire
import json
import os
import sys
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
try:
    from .pd_utils import drop_zero_variance_cols
    from .nb_utils import to_list, descending_sort, remove_prefix, load_json, subset, remove_prefix
except ImportError: # FIXME
    from pd_utils import drop_zero_variance_cols
    from nb_utils import to_list, descending_sort, remove_prefix, load_json, subset, remove_prefix


def infer_metadata(grid_name):

    #x = remove_prefix(grid_name, 'grid.')
    meta = {}
    for entry in grid_name.split('.'):
        if '_' in entry:
            k,v =  entry.split('_')
            meta[k] = v
        else:
            meta[entry] = True
    return meta
    

def read_log_json_file(f, pattern='train_loss') -> dict:
    """ Find a line with train loss in it and try to read it to json."""
    lns = Path(f).open().read().split('\n')
    matched_line = None
    for l in lns:
        if pattern in l:
            matched_line = l
    if matched_line is None:
        raise ValueError(f'none of {len(lns)} lines had the substring train_loss')
    strang = matched_line.split('|')[-1]
    record  = json.loads(strang)
    return record


types = [int, float]

#def trycast(x):

def tryfloat(x):
    if isinstance(x, pd.Series):
        try:
            return x.astype(float)
        except Exception:
            return x
    elif isinstance(x, pd.DataFrame):
        return x.apply(lambda x: tryfloat(x))
    else:
        try: 
            return float(x)
        except TypeError:
            return x


   
def divide_byte_cols(df):
    byte_cols = subset(df, ['bytes']).columns
    df[byte_cols] = df[byte_cols] / 1e9
    return df


def find_read_stderr(dname):
    err_path = list(dname.glob('*stderr*'))
    if not err_path:
        return {'oom': False, 'populated': False}
    content = Path(err_path[0]).open().read()
    return {'oom': 'CUDA out of memory.' in content, 'populated': True}


def make_metadata_table(pattern) -> pd.DataFrame:
    """
    Args:
        pattern: (str) /checkpoint/sshleifer/2020-11-23/*/train.log* (should be quoted on command line)
        csv_path: (str) where to save if suffix is .md will save markdown, otherwise csv.
    """
    records = []
    matches = list(glob(pattern))
    dct = {}
    for f in matches:
        k = Path(f).parent
        dct[k.name] = infer_metadata(k.name)
        dct[k.name].update(find_read_stderr(k))

    tab = pd.DataFrame.from_dict(dct).T
    tab['offload'] = tab['offload'].fillna(0).astype(bool)
    int_cols = ['d', 'ffn']
    tab[int_cols] = tab[int_cols].astype(int)
    tab = tab.fillna(False)
    return tab[tab.populated].drop(['populated', 'bsz', 'ngpu1', 'grid', 'ffn_grid'], 1,errors='ignore')


def make_sweep_table(pattern, csv_path=None, keep_cols=None, sort_col=None, interactive=False, add_metadata=False) -> None:
    """
    Args:
        pattern: (str) /checkpoint/sshleifer/2020-11-23/*/train.log* (should be quoted on command line)
        csv_path: (str) where to save if suffix is .md will save markdown, otherwise csv.
    """
    records = []
    matches = list(glob(pattern))
    for f in matches:
        #import ipdb; ipdb.set_trace()
        #if len(lns)
        try:
            record = read_log_json_file(f)
            record['path'] = Path(f).parent.name
            #record['failed'] = bool(list(Path(f).parent.glob('*stderr*')))
            #record['success'] = 1
            records.append(record)
        except Exception as e:
            # records.append({'path': Path(f).parent.name, 'success': 0})
            print(f'Failed on {f} with {e}')
    if len(records) == 0:
        raise ValueError(f'None of the {len(matches)} log files are ready to be parsed.')
    df = pd.DataFrame(records)
    def assign_treatment(pth):
        if 'ckpt_1.offload_1' in pth: return 3
        elif 'ckpt_1' in pth: return 2
        else: return 1
    df['method'] = df['path'].apply(assign_treatment)
    #import ipdb; ipdb.set_trace()
    # Improvements: remove common prefix, suffix from path
    df = drop_zero_variance_cols(df)
    df = df.set_index('path').pipe(tryfloat).round(2).sort_index()
    df = df.rename(columns=lambda x: remove_prefix(x, 'train_').replace('.', '_'))
    #df.to_markdown(tablefmt="grid")
    #pd.to_numer
    #df.apply(lambda x:)
    if keep_cols is not None:
        df = df[to_list(keep_cols) + ['method']]
    if sort_col is not None:
        df = descending_sort(df, sort_col)

    # gb = df.groupby(['bsz', 'method',])
    # aggd = gb.median()
    # aggd['n_run'] = gb.size()
    # #aggd['n_success'] = gb.success.sum()
    # aggd = aggd.reset_index(['method'])
    aggd = df
    if interactive: return aggd
    
    if csv_path is not None:
        if csv_path.endswith('md'):
            aggd.to_markdown(Path(csv_path).open('w'))
        else:
            aggd.to_csv(csv_path)
    
    print(aggd.to_markdown(tablefmt="grid"))


    
        

def fairseq_log_df(path, pattern='train_inner'):
    """Read train_inner logs (each step)"""
    lns = Path(path).open().read().split('\n')
    records = []
    for l in lns:
        if pattern not in l: continue
        strang = l.split('|')[-1]
        record  = json.loads(strang)
        records.append(record)    
    return pd.DataFrame(records).pipe(tryfloat)


def compare_mem_stats(p1, p2):
    s1 = pd.Series(load_json(p1))
    s2 = pd.Series(load_json(p2))
    increase = s1 - s2
    pct_increase  = (s1 /s2)-1
    tab = pd.DataFrame(dict(p1=s1,p2=s2,increase=increase, pct_increase=pct_increase.round(2)))
    t = subset(tab.T, ['.all.']).T
    return t
# records = {}
# for c in [4,8,16,24]:
#     slc = df[df.bsz==c]
#     deltas = (slc[slc.method==3].iloc[0] / slc[slc.method==2].iloc[0]) - 1
#     records[c]= deltas
    
if __name__ == '__main__':
    #make_sweep_table("/checkpoint/sshleifer/2020-11-23/*/train.log*",  "sweep.md")
    Fire(make_sweep_table)