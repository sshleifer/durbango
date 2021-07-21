import pandas as pd
from .cross_val import cross_val_predict_proba_df
import matplotlib.pyplot as plt


def prepare_train_logs_in_fdf_format(paths_to_all_train_log_files):
    pass

"""
fdf = prepare_train_logs_in_fdf_format(paths_to_all_train_log_files)
# Take in data and forecast
extrapolator = Extrapolator(fdf, loss_col, step_col, bsz_col, id_col, numerical_feature_cols, categorical_feature_cols)

# train model, make forecasts and populate extrapolator.metrics
fdf_with_forecasts = extrapolator.forecast_loss(fdf, up_to_step=200000, predict_ids=None, include_groundtruth=True, forecast_suffix='_forecast')

extrapolator.plot(fdf_with_forecasts, ids=None, ci_percentile=.05)
"""
U='K updates'
KU = 'K updates'
TSEEN = 'tokens_seen_bil'
VC = ['id', 'ts', 'ppl', 'K updates']
WGPU = 'wall_gpu_hours'
FAIRSEQ_NUMERIC_FEATURES = []
FAIRSEQ_CATEGORICAL_FEATURES = []

def select_idlist(df, id_list):
    return df[df['id'].isin(id_list)]
def id_to_series(df, id, metric, idx_col=U):
    return select_idlist(df, [id]).drop_duplicates(idx_col, keep='last').set_index(idx_col)[metric]

def compare_runs(df, ids=None, metric='ppl', idx_col=U):
    if ids is None: ids = df['id'].unique()
    return pd.concat([id_to_series(df, id, metric, idx_col=idx_col).to_frame(id) for id in ids], axis=1)


from sklearn.model_selection import cross_val_predict
import numpy as np
from durbango.cross_val import cross_val_predict_proba_df

def make_xydf(fdf, id_list=None, window=2):
    if id_list is None:
        id_list = fdf.id.unique().tolist()
    xydfs = []
    for eid in id_list:
        cmp = compare_runs(fdf[fdf['id'] == eid], metric='nll_loss', idx_col=KU).drop_duplicates(keep='last').sort_index()
        cmp = cmp.reindex(np.arange(0, cmp.index.max(), step=1))
        ser = cmp[eid]
        xdf = pd.DataFrame(
            dict(
                d1=ser.diff(),
                d5=ser.diff(int(window/2)),
                dw=ser.diff(window),
                second_moment=ser.diff().diff(),
                lev=ser)).shift(window)
        xdf['t'] = xdf.index.values
        xdf['t_squared'] = xdf.t **2
        xdf['log_t'] = xdf.t.pipe(np.log1p)
        # t=ku, log_t=np.log(ku), tsq=ku**2)
        xydfs.append(xdf.assign(y=ser, id=eid).dropna(subset=['y', 'lev']).reset_index(drop=True))
    return pd.concat(xydfs)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def curve_fn(x, a, b, c):
    # monotically descreasing in range (0,300) if -a >= c* np.log(300)*2
    if b > (-a / 11.1) and a < 0:
        b = -a / 11.1
    logx = np.log(x)
    return a * logx + b * logx ** 2 + c
    #return a * np.log(x) + c * np.log(x)**2 + d


def project_many_ids(fdf, ids, **kwargs):
    stats = {}
    plots = {}
    for eid in ids:
        try:
            plots[eid], stats[eid] = run_curve_fit(fdf, eid, **kwargs)
        except RuntimeError:
            print(f'failed for {eid}')
    table = pd.DataFrame(stats).T.astype(float).sort_values('ppl_min').dropna(how='all', axis=1)
    return Viewer(table, plots)


ID = 'id'
def add_is_train_column(xydf, frac_holdout_ids=0.5, min_steps_per_test_id=10):
    # xydf: features should already be built
    max_ku = xydf.groupby(ID)[KU].max()
    splittable_ids = max_ku[max_ku > 10].index
    holdout_ids = np.random.choice(splittable_ids,  size=int(len(splittable_ids) * frac_holdout_ids))
    print(f'holding out {len(holdout_ids)} ids')
    # if KU > min_steps_per_test_id AND id in holdout_ids is_train=False
    enough_steps = xydf[KU] > min_steps_per_test_id
    in_holdout_ids = xydf[ID].isin(holdout_ids)
    xydf['is_train'] = np.logical_not(np.logical_and(enough_steps, in_holdout_ids))
    return xydf # with new_column is_train


from tqdm import tqdm
def rolling_curve_predict(fdf, roll_window=10, tmin=10):
    preds = []
    for i in tqdm(range(tmin, int(fdf[KU].max()), roll_window)):
        slc = fdf[fdf[KU]< i + roll_window]
        ids_to_predict = slc[slc[KU] > i]['id'].unique()
        stats = {}

        for eid in ids_to_predict:
            try:
                (xhat, yhat, _, __, ___, pred_df), stats = run_curve_fit(fdf, eid, trn_cutoff=i, xhat_range=i+roll_window)
                # TODO(SS): use xhat at inference time
                keep_preds = pred_df.loc[(pred_df[KU]> i) & (pred_df[KU] < (i+roll_window)) , [KU, ID, 'yhat']].copy()
                preds.append(keep_preds)
            except (RuntimeError):
                pass
    pred_df = pd.concat(preds).drop_duplicates().rename(columns={'yhat': f'yhat_roll{roll_window}'})
    vc = pred_df.groupby([ID, KU]).size().sort_values(ascending=False)
    if vc.max() > 1:
        print(f'duplicates detected')
        print(vc.head())
    return pred_df



from functools import reduce
import warnings

class Viewer:
    def __init__(self, table, pldata):
        self.table = table
        self.pldata = pldata
        self.ids = table.index.tolist()

    def plot(self, eid):
        xhat, yhat, x, y, trn_cutoff, _ = self.pldata[eid]
        make_plot(xhat, yhat, x, y, eid=eid, trn_cutoff=trn_cutoff)


def diff_filter(slc):
    """return updates where PPL went up"""
    delta = slc.set_index(KU)['nll_loss'].sort_index().diff()
    to_drop = delta[delta > 0].index
    return to_drop

def make_plot(xhat, yhat, x, y, eid='', trn_cutoff=None):
    if trn_cutoff is None:
        trn_cutoff = x.max()
    plt.figure()
    plt.title(eid)
    plt.plot(xhat, yhat, 'r-', label="Fitted Curve")
    plt.plot(x, y, 'ko', label="Original Logs")
    plt.axvline(trn_cutoff)
    plt.legend()

def run_curve_fit(fdf: pd.DataFrame, eid='dense.dl12.d2048.ngpu64', trn_cutoff=100, xhat_range=300, t_min=10):
    KU = 'K updates'
    if trn_cutoff is None:
        trn_cutoff = fdf.loc[fdf['id'] == eid, KU].max()
    #intercept = fdf.loc[(fdf['id'] == eid)].nll_loss.max()

    slc_full = fdf.loc[(fdf['id'] == eid) & (fdf[KU] > t_min)]
    slc_full = slc_full[~slc_full[KU].isin(diff_filter(slc_full))]
    slc_trn = slc_full.loc[(slc_full[KU] <= trn_cutoff)]
    x = slc_trn[KU].values
    yn = slc_trn.nll_loss.values
    if slc_trn.empty:
        raise RuntimeError('empty data')

    with warnings.catch_warnings(record=True):
        popt, pcov = curve_fit(curve_fn, x, yn, p0=np.array([-0.16257,  0.01347,  3.92]),
                           #bounds=([-10, 0, intercept-0.5], [np.inf, np.inf, intercept+0.5])
                           )
    yhat = curve_fn(slc_full[KU].values, *popt)
    pred_df = slc_full.copy().assign(yhat=yhat)
    pred_df['error'] = pred_df['nll_loss'] - pred_df['yhat']
    err_tr = pred_df[pred_df[KU] <= trn_cutoff].error
    err_te = pred_df[pred_df[KU] > trn_cutoff].error

    xhat = np.linspace(t_min, xhat_range, num=(xhat_range-t_min)*10)
    assert xhat[-1] == xhat_range
    yhat = curve_fn(xhat, *popt)
    stats = pd.Series(dict(err_te=err_te.mean(), err_te_abs=err_te.abs().mean(),
                 err_tr=err_tr.mean(), err_tr_abs=err_tr.abs().mean()))
    stats[f'nll_min'] = min(yhat)#[-1]
    stats[f'ppl_min'] = 2**min(yhat)#[-1]
    stats['end_slope'] = 2**(yhat[-1]) - 2**(yhat[-500 if len(yhat) > 500 else 0])
    stats['trn_cutoff'] = trn_cutoff
    for i, p in enumerate(popt):
        stats[f'p_{i}'] = p
    data = (xhat, yhat, slc_full[KU].values, slc_full.nll_loss, trn_cutoff, pred_df)
    return data, stats


merge_cols = ['K updates', 'id']
def make_curve_features(fdf, windows=(10,30,100), tmin=10):
    dfs = [rolling_curve_predict(fdf, roll_window=w, tmin=tmin) for w in windows]
    merged = merge_dfs(dfs, on=merge_cols, how='outer')
    return merged

def merge_dfs(dfs, **kwargs):
    return reduce(lambda left, right: pd.merge(left, right, **kwargs), dfs)




# clf = RidgeCV()
#
# clf.fit(xydf[xdf.columns], xydf['y'])
# pd.Series(dict(zip(xdf.columns.tolist(), clf.coef_))).to_frame('coef_').round(2)
#
#
# xydf['yhat'] = cross_val_predict(clf, xydf[xdf.columns], xydf['y'], cv=10)
# xydf['err'] = xydf['y'] - xydf['yhat']

class Extrapolator:
    def __init__(self, fdf, loss_col='nll_loss', step_col='num_updates', bsz_col='bsz', id_col='id', numerical_feature_cols=FAIRSEQ_NUMERIC_FEATURES, categorical_feature_cols=FAIRSEQ_CATEGORICAL_FEATURES):
        self.fdf = fdf
        self.loss_col = loss_col
        self.step_col = step_col
        self.id_col = id_col
        self.bsz_col = bsz_col
        self.num_features = numerical_feature_cols
        self.cat_features = categorical_feature_cols

        self.xydf = make_xydf(self.fdf)

    def _fit_model(self, xydf, y_col):
        pass


    def forecast_loss(self, up_to_step=200000, predict_ids=None, include_groundtruth=True, forecast_suffix='_forecast') -> pd.DataFrame:
        pass
        return pred_df

    def plot(self, fdf_with_forecasts, ids=None, ci_percentile=.05):
        pass
