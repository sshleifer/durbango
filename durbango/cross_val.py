import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from durbango.nb_utils import shape_assert

def predict_proba_fn_binary(m, x): return m.predict_proba(x)[:, 1]
def predict_proba(m, x): return m.predict_proba(x)
def predict_fn(m, x): return m.predict(x)

predict_fns = {'binary': predict_proba_fn_binary, 'multiclass': predict_proba, 'regression': predict_fn}

def cross_val_predict_proba_df(m, X, y, n_splits=5, task_type='binary', sample_weight=None,
                               stratified=False, random_state=None):
    if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.RangeIndex): raise IndexError(
        'If passing dataframe input, we require RangeIndex. try reset_index(drop=True) before/after creating X'
    )
    shape_assert(X, y)
    predict_func = predict_fns[task_type]
    
    if task_type != 'multiclass':
        preds = pd.Series({k: np.nan for k in X.index})
    else:
        n_classes = len(np.unique(y))
        preds = pd.DataFrame({k: [np.nan] * n_classes for k in X.index}).T
        assert preds.shape == (len(X.index), n_classes)
    if stratified:
        gen = StratifiedKFold(n_splits=n_splits, random_state=random_state).split(X, y)
    else:
        gen = KFold(n_splits=n_splits, shuffle=False, random_state=random_state).split(X)  # Is shuffle=False required

    for train, test in gen:
        if sample_weight is None:
            m.fit(X.iloc[train, :], y.iloc[train])
        else:
            m.fit(X.iloc[train, :], y.iloc[train], sample_weight=sample_weight.iloc[train])
        preds.iloc[test] = predict_func(m, X.iloc[test, :])
    return preds


def cross_val_predict_proba(m, X, y, predict_proba_fn=lambda m, x: m.predict_proba(x),
                            n_splits=5, combine_fn=np.vstack, binary=True, random_state=None):
    """Get out of fold predictions from a classifier. Alignment based on KFold.split. User should shuffle beforehand"""
    if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.RangeIndex): raise TypeError(
        'If passing dataframe input, we require RangeIndex. try reset_index(drop=True) before/after creating X'
    )
    cv = KFold(n_splits=n_splits, shuffle=False, random_state=random_state)
    preds = []
    for train, test in cv.split(X):
        m.fit(X[train, :], y[train])
        preds.append(predict_proba_fn(m, X[test, :]))  # Does this shuffle badly?
    preds = combine_fn(preds)  # cro
    return preds[:, 1] if binary else preds


def avg_predict_proba(clfs, X_test, predict_proba_fn=lambda m, x: m.predict_proba(x)):
    """Ensemble multiple models' predictions"""
    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(X_test)
        else:
            preds += clf.predict_proba(X_test)
    preds = preds / len(clfs)
    return preds
