from sklearn.linear_model import HuberRegressor
from analysis_tools import *


def get_regression_model(X, y, fit_intercept=True):
    return HuberRegressor(
        fit_intercept=fit_intercept,
        epsilon=1.345,
        alpha=0,
        tol=1e-6,
        max_iter=1000,
    ).fit(X, y)


def get_prediction_analysis(y, y_hat):
    # plot_true_vs_pred(y_true=y, y_pred=y_hat)

    return get_squared_errors(y_true=y, y_pred=y_hat), get_qlike(y_true=y, y_pred=y_hat)
