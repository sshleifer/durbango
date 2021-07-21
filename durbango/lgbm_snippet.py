import lightgbm as lgb


feature_cols = ['yhat_roll10', 'yhat_roll30', 'yhat_roll100']
y_col = 'nll_loss'
X_train = fdf3.loc[fdf3.is_train, feature_cols]
X_val = fdf3.loc[~fdf3.is_train, feature_cols]
y_train = fdf3.loc[fdf3.is_train, y_col]
y_val = fdf3.loc[~fdf3.is_train, y_col]


lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    'num_leaves': 5,
    'metric': ['l1', 'l2'],
    'verbose': -1
}

evals_result = {}  # to record eval results for plotting
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=[lgb_train, lgb_test],
                feature_name=feature_cols,
                evals_result=evals_result,
                verbose_eval=10)
