import numpy as np
from utils import get_data, transform_predictors_to_dwm, transform_volatility_by_horizon
from sklearn.linear_model import HuberRegressor
from analysis_tools import get_squared_errors, get_qlike


def get_regression_model(X, y, fit_intercept=True):
    return HuberRegressor(
        fit_intercept=fit_intercept,
        epsilon=1.345,
        alpha=0,
        tol=1e-6,
        max_iter=1000,
    ).fit(X, y)


def get_prediction_loss(y, y_hat):
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

    return get_prediction_loss(y_test, y_hat)


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

    return get_prediction_loss(y_test, y_hat)


def regress(predictors, true_volatility, horizon, estimation_method):
    predictors_transformed = np.array(
        transform_predictors_to_dwm(predictors, horizon)
    ).T
    true_volatility_transformed = np.array(
        transform_volatility_by_horizon(true_volatility, horizon)[22:]
    )

    return estimation_method(predictors_transformed, true_volatility_transformed, 1000)


def assign_loss_values(
    loss_values, regression_results, horizon, estimation_method, model
):
    for loss_name, agg_loss_values in [
        ("squared_error", regression_results[0]),
        ("qlike", regression_results[1]),
    ]:
        loss_values[(horizon, estimation_method, model, loss_name, "mean")] = (
            agg_loss_values[0]
        )
        loss_values[(horizon, estimation_method, model, loss_name, "median")] = (
            agg_loss_values[1]
        )


def standardize_loss(loss_values, by_model):
    # Note: index 2 refers to the model
    for spec, _ in loss_values.items():
        if spec[2] == by_model:
            continue
        loss_values[spec] = (
            loss_values[spec]
            / loss_values[(spec[0], spec[1], by_model, spec[3], spec[4])]
        )
    for spec, _ in loss_values.items():
        if spec[2] == by_model:
            loss_values[spec] = 1


def get_regression_results(lr_predictors, target, horizons, estimation_methods, data):
    loss_values = dict()

    for horizon in horizons:
        for estimation_method in estimation_methods:
            estimation_function = (
                rolling_window if estimation_method == "rw" else increasing_window
            )
            for model, predictors in lr_predictors.items():
                regression_results = regress(
                    data[predictors], data[target], horizon, estimation_function
                )
                assign_loss_values(
                    loss_values, regression_results, horizon, estimation_method, model
                )

                print(
                    f"Finished results for horizon {horizon}, "
                    + f"estimation method {estimation_method}, model {model}"
                )

    standardize_loss(loss_values, "HAR-TV")

    return loss_values
