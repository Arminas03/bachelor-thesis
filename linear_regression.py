from sklearn.linear_model import HuberRegressor
from analysis_tools import *
import statsmodels.api as sm


def get_regression_model(X, y, fit_intercept=True):
    return HuberRegressor(
        fit_intercept=fit_intercept,
        epsilon=1.345,
        alpha=0,
        tol=1e-6,
        max_iter=1000,
    ).fit(X, y)


# def get_regression_model(X, y, fit_intercept=True):
#     if fit_intercept:
#         X = sm.add_constant(X, has_constant="add")
#     return sm.RLM(y, X, M=sm.robust.norms.HuberT(t=1.345)).fit(maxiter=1000, tol=1e-6)


def get_prediction_analysis(y, y_hat):
    # plot_true_vs_pred(y_true=y, y_pred=y_hat)
    mse = get_mse(y_true=y, y_pred=y_hat)
    qlike = get_qlike(y_true=y, y_pred=y_hat)
    print(f"MSE: {mse}")
    print(f"QLIKE: {qlike}")

    return mse, qlike
