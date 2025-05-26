import numpy as np
import pandas as pd

from sklearn.linear_model import HuberRegressor
from analysis_tools import get_squared_errors, get_qlike
from utils import transform_predictors_to_dwm, transform_volatility_by_horizon


def get_regression_model(X, y, alpha_ridge=0, fit_intercept=True):
    return HuberRegressor(
        fit_intercept=fit_intercept,
        epsilon=1.345,
        alpha=alpha_ridge,
        tol=1e-6,
        max_iter=2000,
    ).fit(X, y)


def get_prediction_loss(y: np.array, y_hat: np.array):
    # plot_true_vs_pred(y_true=y, y_pred=y_hat)

    return get_squared_errors(y_true=y, y_pred=y_hat), get_qlike(y_true=y, y_pred=y_hat)


def winsorise(y, y_hat):
    if y_hat > max(y):
        return max(y)
    if y_hat < min(y):
        return min(y)
    return y_hat


def increasing_window(X, y, initial_iw_size=1000):
    X_train = X[:initial_iw_size]
    y_train, y_test = y[:initial_iw_size], y[initial_iw_size:]
    y_hat = []

    for i in range(initial_iw_size, len(y)):
        y_hat.append(
            get_regression_model(X_train, y_train).predict(X[i].reshape(1, -1))[0]
        )
        y_hat[-1] = winsorise(y_train, y_hat[-1])
        X_train = np.append(X_train, X[i].reshape(1, -1), axis=0)
        y_train = np.append(y_train, [y[i]], axis=0)

    return y_test, y_hat


def rolling_window(X, y, rw_size=1000):
    X_train = X[:rw_size]
    y_train, y_test = y[:rw_size], y[rw_size:]
    y_hat = []

    for i in range(rw_size, len(y)):
        y_hat.append(
            get_regression_model(X_train, y_train).predict(X[i].reshape(1, -1))[0]
        )

        y_hat[-1] = winsorise(y_train, y_hat[-1])
        X_train = np.append(X_train, X[i].reshape(1, -1), axis=0)[1:]
        y_train = np.append(y_train, [y[i]], axis=0)[1:]

    return y_test, y_hat


def regress(
    predictors: pd.DataFrame, true_volatility: pd.Series, horizon, estimation_method
):
    """
    Note that predictors and true volatility must be of equal size and
    aligned date-by-date. The transformation is done within this function.
    Returns y_test, y_hat np arrays
    """
    predictors_transformed = np.array(
        transform_predictors_to_dwm(predictors, horizon)
    ).T
    true_volatility_transformed = np.array(
        # 22 observations lost due to monthly, push by 1 due to future
        transform_volatility_by_horizon(true_volatility, horizon)[21 + 1 :]
    )

    return estimation_method(predictors_transformed, true_volatility_transformed, 1000)
