import numpy as np


def output_df_scorer(df, y_col='y', yhat_col='yhat', sample_weight_col=None):
    '''Compute some metrics on a df with y and yhat columns'''
    if sample_weight_col is None:
        sample_weights = None
    else:
        sample_weights = df[sample_weight_col]
    ser = pd.Series({
        'r2': r2_score(df[y_col], df[yhat_col], sample_weight=sample_weights),
        # 'gini': calc_normalized_gini(df[y_col], df[yhat_col]),
        'n': df.shape[0],
        'pct_large_yhat': (df[yhat_col].abs() > 0.2).mean(),
        'mn_yhat': df[yhat_col].mean(),
        'sign_agree': ((df[yhat_col] > 0) == (df[y_col] > 0)).mean(),
        'sign_agree_non_zero': ((df[yhat_col] > 0) == (df[y_col] > 0)).loc[df[y_col] != 0].mean(),
        'corr': df[yhat_col].corr(df[y_col]),
        'mse': mean_squared_error(df[y_col], df[yhat_col], sample_weight=sample_weights),
        'mae': mean_absolute_error(df[y_col], df[yhat_col], sample_weight=sample_weights)
    })
    ser['rmse'] = np.sqrt(ser['mse'])
    return ser


def output_df_scorer_classification(df, y_col='y', yhat_col='yhat'):
    '''Usually applied on a groupby, outputs score metrics.'''
    return pd.Series({
        'n': df.shape[0],
        'auc': safe_roc_auc_score(df[y_col], df[yhat_col]),
        'mean_y': df[y_col].mean(),
        'f1_score': proba_f1_score(df[y_col], df[yhat_col]),
        'accuracy': proba_accuracy_score(df[y_col], df[yhat_col]),
        'precision': proba_precision_score(df[y_col], df[yhat_col]),
        'recall': proba_recall_score(df[y_col], df[yhat_col]),
        'mean_yhat': df[yhat_col].mean(),
        'logloss': log_loss(df[y_col], df[yhat_col])
    })


def score_output_df_and_log_n_dropped(odf, sample_weight_col=None):
    '''Run output_df_scorer on a supplied xydf'''
    scores = output_df_scorer(odf.dropna(subset=['y', 'yhat']),
                              sample_weight_col=sample_weight_col)
    scores.loc['n_dropped'] = odf.shape[0] - scores.loc['n']
    return scores


def make_score_table(output_df_tr, output_df_val, scorer=output_df_scorer):
    '''helper to run output_df_scorer on test and train'''
    tab = scorer(output_df_tr).to_frame('train').assign(valid=scorer(output_df_val))
    return tab


def make_period_score_table(output_df, freq, scorer=output_df_scorer, **scorer_kwargs):
    '''Apply output df scorer to each period in output_df for examining backtest performance

    Args:
        output_df: (pd.DataFrame) indexed by date
        freq: (str) period string, consistent with Pandas notation (e.g. A, Q, M, etc.)
        scorer: (func) output_df_scorer / output_df_scorer_classification
        **scorer_kwargs: passed to scorer func

    Return:
        (pd.DataFrame) with scores where each row represents metrics for a given period
    '''
    grouper = pd.Grouper(key='date', freq=freq)
    score_df = output_df.reset_index().groupby(grouper).apply(scorer, **scorer_kwargs)
    return score_df


def make_score_table_classification(cv):
    '''helper to run output_df_scorer on test and train'''
    return make_score_table(cv, output_df_scorer_classification)


def score_metrics_by_size(pred_df, threshold, col_name):
    '''Score metrics based on signal (or return) size'''
    above_threshold_mask = np.abs(pred_df[col_name]) > threshold

    above_threshold_metrics = (
        output_df_scorer(pred_df[above_threshold_mask]).to_frame('above threshold metrics')
    )
    below_threshold_metrics = (
        output_df_scorer(pred_df[~above_threshold_mask]).to_frame('below threshold metrics')
    )
    metrics_df = pd.concat([above_threshold_metrics, below_threshold_metrics], axis=1)
    return metrics_df
