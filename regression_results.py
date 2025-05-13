import pandas as pd
import numpy as np
from linear_regression import get_regression_model, get_prediction_analysis


def get_data(path):
    data = pd.read_csv(path).rename(columns={"???Date": "date"})
    data["date"] = pd.to_datetime(data["date"])

    return data


def transform_volatility_by_horizon(true_volatility, horizon):
    transformed_vol = []
    c = 1
    curr_sum_vol = 0
    for i, true_vol in enumerate(true_volatility):
        curr_sum_vol += true_vol
        if c < horizon:
            c += 1
            continue

        transformed_vol.append(curr_sum_vol)

        curr_sum_vol -= true_volatility.iloc[i - horizon + 1]

    return transformed_vol


def transform_predictors_to_dwm(predictors, horizon):
    final_predictors = []
    for _, predictor in predictors.items():
        for steps in [1, 5, 22]:
            final_predictors.append(
                transform_volatility_by_horizon(predictor, steps)[
                    (22 - steps) : -horizon
                ]
            )

    return final_predictors


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

    return get_prediction_analysis(y_test, y_hat)


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

    return get_prediction_analysis(y_test, y_hat)


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


def get_regression_results(lr_predictors, target, horizons, estimation_methods):
    loss_values = dict()
    data = get_data("data_files/Todorov-Zhang-JAE-2021.csv")

    for horizon in horizons:
        for estimation_method in estimation_methods:
            estimation_function = (
                rolling_window if estimation_method == "rw" else increasing_window
            )
            for model, predictors in lr_predictors.items():
                print(
                    f"Finished results for horizon {horizon}, "
                    + f"estimation method {estimation_method}, model {model}"
                )

                regression_results = regress(
                    data[predictors], data[target], horizon, estimation_function
                )
                assign_loss_values(
                    loss_values, regression_results, horizon, estimation_method, model
                )

    standardize_loss(loss_values, "HAR-TV")

    return loss_values
