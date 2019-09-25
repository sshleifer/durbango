"""For Gradient Boosted Tree/LightGBM Style models. Please do not add dependencies to global scope."""
def shap_plot(my_model, val_X, class_num=1):
    """Create object that can calculate shap values"""
    import shap  # package used to calculate Shap values
    explainer = shap.TreeExplainer(my_model)

     # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(val_X)
    if class_num is not None:
        shap_values = shap_values[class_num]

    # Make plot. Index of [1] is explained in text below.
    shap.summary_plot(shap_values, val_X)


def permutation_importance(my_model, val_X, val_y, ret_df=False):
    import eli5
    from eli5.sklearn import PermutationImportance
    perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
    if ret_df:
        return eli5.explain_weights_df(perm, feature_names=val_X.columns.tolist())
    else:
        return eli5.show_weights(perm, feature_names=val_X.columns.tolist())
