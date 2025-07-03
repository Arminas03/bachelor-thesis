import numpy as np
import pandas as pd
from typing import Callable

from sklearn.linear_model import HuberRegressor
from utils import (
    transform_predictors_to_dwm,
    transform_volatility_by_horizon,
    get_squared_errors,
    get_qlike,
)


def get_regression_model(
    X: np.array, y: np.array, alpha_ridge: float = 0, fit_intercept: bool = True
) -> HuberRegressor:
    """
    Fits and returns Huber regression model
    """
    return HuberRegressor(
        fit_intercept=fit_intercept,
        epsilon=1.345,
        alpha=alpha_ridge,
        tol=1e-6,
        max_iter=2000,
    ).fit(X, y)


def get_prediction_loss(
    y: np.array, y_hat: np.array
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Returns mean and median loss values
    """
    # plot_true_vs_pred(y_true=y, y_pred=y_hat)

    return get_squared_errors(y_true=y, y_pred=y_hat), get_qlike(y_true=y, y_pred=y_hat)


def winsorise(y: np.array, y_hat: float) -> float:
    """
    Insanity filter of Todorov and Zhang (2022)
    """
    if y_hat > max(y):
        return max(y)
    if y_hat < min(y):
        return min(y)
    return y_hat


def increasing_window(
    X: np.array, y: np.array, initial_iw_size: int = 1000
) -> tuple[np.array, np.array]:
    """
    Performs increasing window estimation. Returns resulting
    predictions of given y along with true (test) values
    """
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

    return y_test, np.array(y_hat)


def rolling_window(
    X: np.array, y: np.array, rw_size: int = 1000
) -> tuple[np.array, np.array]:
    """
    Performs rolling window estimation. Returns resulting
    predictions of given y along with true (test) values
    """
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

    return y_test, np.array(y_hat)


def regress(
    predictors: pd.DataFrame,
    true_volatility: pd.Series,
    horizon: int,
    estimation_method: Callable[[np.array, np.array, int], tuple[np.array, np.array]],
    regress_log: bool = False,
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

    if regress_log:
        predictors_transformed = np.log(predictors_transformed)
        true_volatility_transformed = np.log(true_volatility_transformed)

    return estimation_method(predictors_transformed, true_volatility_transformed, 1000)
