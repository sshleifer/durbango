import numpy as np
import pandas as pd
import datetime
import warnings
from durbango.nb_utils import remove_prefix, tqdm_nice
from fb_sweep.agg_results import find_last_matching_line, reverse_readline, find_all_matching_lines, tryfloat
from pathlib import Path
import torch
import json
U='K updates'
KU = 'K updates'
TSEEN = 'tokens_seen_bil'
VC = ['id', 'ts', 'ppl', 'K updates']
WGPU = 'wall_gpu_hours'
import matplotlib.pyplot as plt
import re
def parse_epoch_logs(paths):
    from fb_sweep.agg_results import find_last_matching_line, reverse_readline
    keep_keys = ['wps', 'ups', 'wpb', 'bsz', 'num_updates', 'train_wall', 'gb_free', 'wall', 'epoch', 'lr']
    int_cols = ['wps', 'wpb', 'bsz', 'num_updates', 'wall', 'epoch']
    res = []
    for p in paths:
        lns = reverse_readline(p + '/train.log')
        try:
            last_inner = find_last_matching_line(lns, 'train_wps')
        except ValueError:
            continue
        ser = pd.Series({remove_prefix(k, 'train_'): v for k, v in last_inner.items() if
                         remove_prefix(k, 'train_') in keep_keys})  #.rename(index={remove_prefix(k, 'train_')})
        ser['path'] = p
        res.append(ser)
    train_log_df = pd.DataFrame(res).pipe(tryfloat)
    train_log_df[int_cols] = train_log_df[int_cols].astype(int)
    #train_log_df['tokens_seen_bil'] = (train_log_df.wps * train_log_df.wall) / 1e9
    train_log_df['wall'] = (train_log_df['num_updates'] / train_log_df['ups'])
    train_log_df['tokens_seen_bil'] = (train_log_df.wps * train_log_df.wall) / 1e9
    train_log_df['id'] = train_log_df.path.apply(lambda x: Path(x).name)
    return train_log_df

def select_idlist(df, id_list):
    return df[df['id'].isin(id_list)]

def select_ids_by_pattern(df, pattern):
    return df.loc[df['id'].str.contains(pattern)]

def parse_entry(ln):
    pth, rest = ln.split('/train.log:')
    record = json.loads(rest.split('|')[-1])
    record['path'] = pth
    record['id'] = pth.split('/')[-1]
    splat = rest.split('|')[0]#.split(':')[1:]
    ts = re.sub('^(\d+):', '', splat).strip()
    record['ts'] = ts#f"{year}{splat.split('2021')[1]}"
    return record



def id_to_series(df, id, metric, idx_col=U):
    return select_idlist(df, [id]).drop_duplicates(idx_col, keep='last').set_index(idx_col)[metric]

def compare_runs(df, ids=None, metric='ppl', idx_col=U, extra_ids=None):
    if ids is None: ids = df['id'].unique().tolist()
    if isinstance(ids, np.ndarray): ids = ids.tolist()
    if isinstance(extra_ids, str):
        ids.append(extra_ids)
    elif isinstance(extra_ids, list):
        ids.extend(extra_ids)
    return pd.concat([id_to_series(df, id, metric, idx_col=idx_col).to_frame(id) for id in ids], axis=1)




def compare_runs(df, ids=None, metric='ppl', idx_col=U, extra_ids=None):
    if ids is None: ids = df['id'].unique().tolist()
    if isinstance(ids, np.ndarray): ids = ids.tolist()
    if isinstance(extra_ids, str):
        ids.append(extra_ids)
    elif isinstance(extra_ids, list):
        ids.extend(extra_ids)
    return pd.concat([id_to_series(df, id, metric, idx_col=idx_col).to_frame(id) for id in ids], axis=1)


def get_train_hours_without_interrupt(df):
    gb = df.groupby('id')
    #g = gb.get_group('shru_baseline.dl12.d2048.moe_w0.01.ngpu64')
    return gb.apply(_get_hours)

def _get_hours(g):
    deltas = g.sort_values('ts').ts.diff() / np.timedelta64(1, 'h')
    med_delta = deltas.median()
    return (deltas[deltas < med_delta * 10]).sum()


def filter_to_training_before_best_ts(g):
    g = g.sort_values('ts')
    best_ts = g[g.ppl == g.ppl.min()].ts.iloc[0]
    return g[g.ts <= best_ts]

def is_es_column(c):
    if c.startswith('SS_'): return True
    if c.startswith('X_'): return True
    return False

def filter_to_training_before_best_ts(g):
    g = g.sort_values('ts')
    best_ts = g[g.ppl == g.ppl.min()].ts.iloc[0]
    g['delta'] = g.ts.diff() / np.timedelta64(1, 'h')
    med_delta = g.delta.median()
    g.loc[g.delta > med_delta, 'delta'] = med_delta  # restarts -> median cost
    g['wall'] = g.delta.fillna(0).cumsum()
    cut_td = g.ts.max() - best_ts
    cut_hrs = cut_td / np.timedelta64(1, 'h')
    g['improving'] =  g.ts <= best_ts
    #best =  g[g.ts <= best_ts].drop('delta', 1)
    best = g.drop('delta', 1)
    return best

def filter_df(df):
    """adds new wall_col. 28s, mostly on concat. Apply takes 5x longer."""
    df = df.sort_values('ts')
    res = []
    cut_info = {}
    for id in tqdm_nice(df['id'].unique()):
        g = df[df['id'] == id]
        filtered = filter_to_training_before_best_ts(g)
        cut_info[id] = g.num_updates.max() - filtered.num_updates.max()
        res.append(filtered.reset_index(drop=True))
    fdf = pd.concat(res)
    right_df = pd.Series(cut_info).to_frame('updates_cut')
    fdf = fdf.merge(right_df, left_on='id', right_index=True, how='left')
    return fdf



def parse_epoch_logs(paths):
    keep_keys = ['wps', 'ups', 'wpb', 'bsz', 'num_updates', 'train_wall', 'gb_free', 'wall', 'epoch', 'lr']
    int_cols = ['wps', 'wpb', 'bsz', 'num_updates', 'wall', 'epoch']
    res = []
    for p in paths:
        lns = reverse_readline(p + '/train.log')
        try:
            last_inner = find_last_matching_line(lns, 'train_wps')
        except ValueError:
            continue
        ser = pd.Series({remove_prefix(k, 'train_'): v for k, v in last_inner.items() if
                         remove_prefix(k, 'train_') in keep_keys})  #.rename(index={remove_prefix(k, 'train_')})
        ser['path'] = p
        res.append(ser)
    train_log_df = pd.DataFrame(res).pipe(tryfloat)
    train_log_df[int_cols] = train_log_df[int_cols].astype(int)
    #train_log_df['tokens_seen_bil'] = (train_log_df.wps * train_log_df.wall) / 1e9
    train_log_df['wall'] = (train_log_df['num_updates'] / train_log_df['ups'])
    train_log_df['tokens_seen_bil'] = (train_log_df.wps * train_log_df.wall) / 1e9
    train_log_df['id'] = train_log_df.path.apply(lambda x: Path(x).name)
    return train_log_df


def parse_all_epoch_logs(paths):
    from fb_sweep.agg_results import find_last_matching_line, reverse_readline, find_all_matching_lines
    keep_keys = ['wps', 'ups', 'wpb', 'bsz', 'num_updates', 'train_wall', 'gb_free', 'wall', 'epoch', 'ppl']
    int_cols = ['wps', 'wpb', 'bsz', 'num_updates', 'wall', 'epoch']
    res = []
    for p in paths:
        lns = reverse_readline(p + '/train.log')
        try:
            last_inner = find_all_matching_lines(lns, 'train_wps')
            last_inner['path'] = p
        except ValueError:
            continue
        res.append(last_inner)
    train_log_df = pd.concat(res).pipe(tryfloat).rename(columns=lambda x: remove_prefix(x, 'train_'))
    #print(train_log_df.columns)
    train_log_df[int_cols] = train_log_df[int_cols].astype(int)
    train_log_df['tokens_seen_bil'] = (train_log_df.wps * train_log_df.wall) / 1e9
    train_log_df['other_wall'] = (train_log_df['num_updates'] / train_log_df['ups'])
    train_log_df['id'] = train_log_df.path.apply(lambda x: Path(x).name)
    train_log_df[KU] = train_log_df.num_updates/1000
    train_log_df['ngpu'] = train_log_df['id'].apply(lambda x: x.split('ngpu')[-1]).astype(int)
    return train_log_df

PREFIX_TO_TEACHER = {
    'gv6': 1181.34 + 1986.61, # gv4_best 1181.34 + (m2m_32_best_16/start.pt) (from shru_longer = 1986.61)
    'gv2': 1986.61, #(m2m_32_best_16/start.pt) (from shru_longer = 1986.61)
    'gv7_80K':  1079, # fdf[fdf['id']=='shru_a100_longer.dl12.d2048.moe_w0.01.ngpu32'].loc[lambda x: x[KU]==80, WGPU]
    'gv7_100k':  1353,
    'm2m': 1957.12, # shru_baseline.dl12.d2048.moe_w0.01.ngpu64  (Note v100 issue)
    'm2d': 1957.12, # shru_baseline.dl12.d2048.moe_w0.01.ngpu64
    'dense': 0,
    'shru': 0,
}

# Scaling law plots

SHRU_IDS = ['shru_baseline.dl12.d2048.moe_w0.01.ngpu64',
 'shru_baseline_es.dl12.d2048.moe_wt0.01.ngpu64',
 'shru_baseline_es_v2.dl12.d2048.moe_w0.01.ngpu64',
 'shru_baseline.dl12.d2048.moe_w0.01.ngpu32',
 'shru_v0.dl12.d2048.moe_w0.01.ngpu32',
 'shru_a100.dl12.d2048.moe_w0.01.ngpu8',
 'shru_a100.dl12.d2048.ngpu16',
 'shru_a100.dl12.d2048.ngpu32',
 'shru_a100_longer.dl12.d2048.moe_w0.01.ngpu32',
 'shru_a100.dl12.d2048.moe_w0.01.ngpu16',
 'shru_a100_v0.dl12.d2048.moe_w0.01.ngpu16',
 'shru_a100_longer.dl12.d2048.ngpu4',
 'shru_a100_longer_first_run.dl12.d2048.ngpu8',
 'shru_a100_longer.dl12.d2048.ngpu8',
 'shru_a100_longer.dl12.d2048.ngpu16',
 'shru_a100_v3.dl12.d2048.ngpu32',
 'shru_v100_v3.dl12.d2048.ngpu32']


scaling_law_ids = ['shru_a100.dl12.d2048.moe_w0.01.ngpu16', 'shru_a100_longer.dl12.d2048.moe_w0.01.ngpu32', 'shru_baseline.dl12.d2048.moe_w0.01.ngpu64', 'shru_a100_longer.dl12.d2048.ngpu8']
def id_to_gpu_int(id):
    return int(id.split('.ngpu')[-1])
def plotter(ids=scaling_law_ids, idx_col=KU, rename_with_ngpu=True, **pl_kw):
    if isinstance(ids, str): ids = [ids]
    pldf = compare_runs(dd[dd.ppl <=18], ids=ids, idx_col=idx_col).sort_index()
    if rename_with_ngpu:
        pldf = pldf.rename(columns=id_to_gpu_int)
    ax = pldf.plot(style='-o',  **pl_kw)
    ax.set_title(idx_col)
    return ax
XCOLS = [KU, TSEEN, WGPU, ]

def triptic(ids=scaling_law_ids):
    plt.figure();
    fig, axs = plt.subplots(1, len(XCOLS), figsize=(20,6));
    for i,c in enumerate(XCOLS):
        plotter(c, ids=ids, ax=axs.flat[i], fontsize=14)
    for ax in axs.flat:
        ax.label_outer()


renamer = {
    'dense.dl12.d2048.ngpu64': 'baseline',
    'd2d.t_355M_gpt3_setting.a0.75.temp1.0.dl12.d2048.ngpu64': 'kd.355M_Teacher',
    'd2d.t_1.3B_gpt3_setting.a0.75.temp1.0.dl12.d2048.ngpu64': 'kd.1.3B_Teacher',
    'd2d.t_2.7B_gpt3_setting.a0.75.temp1.0.dl12.d2048.ngpu64': 'kd.2.7B_Teacher',
    'm2d.t_moe_64e_longer_170K.a0.75.temp1.0.dl12.d2048.ngpu64': 'kd.64e_Teacher',
    'm2d_wt_rand.t_moe_16e_longer.a0.75.temp1.0.dl12.d2048.ngpu16': 'kd.16e_Teacher',
}


ad2 = {'ppl': 'min', KU: 'max', 'ts': 'max', 'ngpu': 'max', 'dl': 'max', 'soft_loss': 'min', 'hard_loss': 'min', 'last_wps': 'median', TSEEN: 'max', WGPU: 'max'}
def make_tab(df, agg_dict=ad2):
    real_agg = {k:v for k,v in agg_dict.items() if k in df.columns}
    tab = df.groupby('id').agg(real_agg).sort_values('ppl').rename(columns={'last_wps': 'wps'})
    #tab['start_ts'] = df.groupby('id').ts.min()
    #tab['hours'] = (tab.ts-tab.start_ts)/np.timedelta64(1, 'h')
    tab['hours'] = get_train_hours_without_interrupt(df).sort_values(ascending=False).round(1)
    tab['wps'] =(tab['wps'].fillna(0) / 1000).astype(int)
    return tab

from pathlib import Path
def make_fdf_and_tab(path='/private/home/sshleifer/fairseq-py/distill_valid_loss_entries.log'):
    records = []
    for ln in Path(path).open().readlines():
        try:
            records.append(parse_entry(ln))
        except Exception as e:
            print(e)
            continue

    df = pd.DataFrame(records).pipe(tryfloat).rename(columns=lambda x: remove_prefix(x, 'valid_'))
    df = df[df.ts != '2021:']
    df = df[df.ts != '20212:']
    df['ts'] = pd.to_datetime(df.ts, errors='coerce')
    df[KU] = df['num_updates'].astype(int) / 1000

    # alloc_cols = df.pipe(subset, ['SS']).columns
    # print(f'last_log: {df.ts.max()}')
    # df = df.dsort('ts')

    def get_dl(x):
        splat = [int(remove_prefix(x, 'dl')) for x in x.split('.') if x.startswith('dl') and not x.startswith('dld')]
        if splat:
            return splat[0]
        else:
            return -1

    df['dl'] = df.id.apply(get_dl)
    df['ngpu'] = df['id'].apply(lambda x: x.split('ngpu')[-1]).astype(int)
    if 'nll_loss' in df.columns:
        df['nll_loss'] = df.nll_loss.fillna(df.hard_loss).fillna(df.loss)
    else:
        df['nll_loss'] = df['loss']
    df['ppl'] = 2 ** df.nll_loss

    # %%capture
    with warnings.catch_warnings(record=True):
        train_log_df = parse_epoch_logs(df.path.unique())
        speed_meta = train_log_df.set_index('path')[[TSEEN, 'num_updates', 'wall', 'wps']].add_prefix(
            'last_').reset_index()
        fdf1 = filter_df(df)
        fdf = fdf1.merge(speed_meta, on='path', how='left')
        fdf[TSEEN] = ((fdf.num_updates / (fdf.last_num_updates - fdf.updates_cut) * fdf[f'last_{TSEEN}']))
        assert fdf.updates_cut.max() == 0.  # not cutting
        fdf[WGPU] = fdf['wall'] * fdf['ngpu']
        if 'soft_loss' in df.columns:
            msk = fdf[
                      'id'] == 'scale_soft_loss10x.ebs512.lr0.0003.t_1.3B_gpt3_setting.a0.75.temp1.0.dl12.d2048.dr0.0.ngpu32'
            fdf.loc[msk, 'soft_loss'] = fdf.loc[msk, 'soft_loss'] / 10.
            msk = fdf['id'] == 'scale_soft_loss5x.ebs512.lr0.0003.t_1.3B_gpt3_setting.a0.75.temp1.0.dl12.d2048.dr0.0.ngpu32'
            fdf.loc[msk, 'soft_loss'] = fdf.loc[msk, 'soft_loss'] / 5.
        # tab = make_tab(fdf[fdf[KU]<200])

        tab = make_tab(fdf)
        tab = tab[tab.ppl < 200]
        #train_df2 = parse_all_epoch_logs(df.path.unique())
    alloc_cols = [x for x in fdf.columns if is_es_column(x)]
    fdf = fdf.drop(alloc_cols, 1)
    return fdf, tab#, train_df2
