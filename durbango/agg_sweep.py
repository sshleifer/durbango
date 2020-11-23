from pathlib import Path
from glob import glob
import pandas as pd
from fire import Fire
import json
from pd_utils import drop_zero_variance_cols
from nb_utils import to_list, descending_sort, remove_prefix


def read_log_json_file(f) -> dict:
    lns = Path(f).open().read().split('\n')
    strang = lns[-3].split('|')[-1]
    record  = json.loads(strang)
    return record

def make_sweep_table(pattern, csv_path=None, keep_cols=None, sort_col=None) -> None:
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
            records.append(record)
        except json.JSONDecodeError:
            print(f'Failed on {f}')
    if len(records) == 0:
        raise ValueError(f'None of the {len(matches)} log files are ready to be parsed.')
    df = pd.DataFrame(records)
    #import ipdb; ipdb.set_trace()
    # Improvements: remove common prefix, suffix from path
    df = drop_zero_variance_cols(df)
    df = df.set_index('path').apply(pd.to_numeric).round(2).sort_index()
    df = df.rename(columns=lambda x: remove_prefix(x, 'train_'))
    #df.to_markdown(tablefmt="grid")
    #pd.to_numer
    #df.apply(lambda x:)
    if keep_cols is not None:
        df = df[to_list(keep_cols)]
    if sort_col is not None:
        df = descending_sort(df, sort_col)
    
    if csv_path is not None:
        if csv_path.endswith('md'):
            df.to_markdown(Path(csv_path).open('w'))
        else:
            df.to_csv(csv_path)
    
    print(df.to_markdown(tablefmt="grid"))

    
    

    
if __name__ == '__main__':
    #make_sweep_table("/checkpoint/sshleifer/2020-11-23/*/train.log*",  "sweep.md")
    Fire(make_sweep_table)