import lightgbm as lgbm
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import NotFittedError


reg_params = dict(objective='regression',
                  metric='l2_root')
unused = {
    'objective': 'regression', 'boosting_type': 'gbdt',
    'verbosity': 1, 'metric': 'l2_root',
}


class LGBMKFold(RegressorMixin):
    default_params = reg_params.copy()

    def __init__(self, **kwargs):
        '''Set up params like num_leaves, google lightgbm parameters for more'''
        self.params = self.default_params
        self.params.update(**kwargs)
        self.is_fitted = False

        self.n_features = None
        self.fitted_regressors = []
        self.kf = None
        self.feature_importances_ = None

    def fit(self, X, y, num_boost_round=1000, n_folds=5, shuffle=True, random_state=128,
            verbose_eval=100, early_stopping_rounds=100):
        '''for each fold in nfolds, hold it out and train a model on the rest of the data'''
        # TODO: make oof preds
        self.n_features = X.shape[1]
        kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

        self.kf = kfold.split(X)
        for _, (train_fold, test_fold) in enumerate(self.kf):
            X_tr, X_val, y_tr, y_val = (X[train_fold, :], X[test_fold, :],
                                        y[train_fold], y[test_fold])
            dtrain = lgbm.Dataset(X_tr, y_tr)
            dvalid = lgbm.Dataset(X_val, y_val)
            bst = lgbm.train(
                self.params, dtrain, num_boost_round,
                valid_sets=dvalid, verbose_eval=verbose_eval,
                early_stopping_rounds=early_stopping_rounds
            )
            self.fitted_regressors.append(bst)
        self.is_fitted = True
        self.feature_importances_ = (
            np.array([reg.feature_importance() for reg in self.fitted_regressors]).mean(0)
        )
        return self

    def predict(self, X):
        '''make predictions by taking mean of all K estimates.'''
        if not self.is_fitted:
            raise NotFittedError()
        preds = [reg.predict(X) for reg in self.fitted_regressors]
        return np.array(preds).mean(0)
