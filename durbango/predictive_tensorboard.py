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

def make_xydf(fdf, id_list=None, window=40):
    if id_list is None:
        id_list = fdf.id.unique().tolist()
    xydfs = []
    for eid in id_list:
        cmp = compare_runs(fdf[fdf['id'] == eid], metric='nll_loss', idx_col=KU).drop_duplicates(keep='last').sort_index()
        cmp = cmp.reindex(np.arange(0, cmp.index.max(), step=0.25))
        ser = cmp[eid]
        xdf = pd.DataFrame(
            dict(
                d1=ser.diff(),
                d5=ser.diff(5),
                d10=ser.diff(10),
                second_moment=ser.diff().diff(),
                lev=ser)).shift(window)
        xdf['t'] = xdf.index.values
        xdf['t_squared'] = xdf.t **2
        xdf['log_t'] = xdf.t.pipe(np.log1p)
        # t=ku, log_t=np.log(ku), tsq=ku**2)
        xydfs.append(xdf.assign(y=ser, id=eid).dropna())
    return pd.concat(xydfs)



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
