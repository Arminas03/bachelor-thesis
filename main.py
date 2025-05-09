import pandas as pd
import numpy as np
from linear_regression import *


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


def increasing_window_estimation(X, y, initial_iw_size=1000):
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


def rolling_window_estimation(X, y, rw_size=1000):
    X_train = X[:rw_size]
    y_train, y_test = y[:rw_size], y[rw_size:]
    y_hat = []

    for i in range(rw_size, len(y)):
        y_hat.append(
            get_regression_model(X_train, y_train).predict(X[i].reshape(1, -1))[0]
        )
        # x_i = X[i].reshape(1, -1)
        # x_i = sm.add_constant(x_i, has_constant="add")
        # y_hat.append(get_regression_model(X_train, y_train).predict(x_i)[0])

        y_hat[-1] = winsorise(y_train, y_hat[-1])
        X_train = np.append(X_train, X[i].reshape(1, -1), axis=0)[1:]
        y_train = np.append(y_train, [y[i]], axis=0)[1:]

    return get_prediction_analysis(y_test, y_hat)


def regress(predictors, true_volatility, horizon):
    predictors_transformed = np.array(
        transform_predictors_to_dwm(predictors, horizon)
    ).T
    true_volatility_transformed = np.array(
        transform_volatility_by_horizon(true_volatility, horizon)[22:]
    )

    return rolling_window_estimation(
        predictors_transformed, true_volatility_transformed, 1000
    )


def main():
    lr_predictors = {
        "HAR-RV": ["RV"],
        "HAR-TV": ["TV"],
        # "HAR-OV": ["OV"],
        # "HAR-EV": ["EV"],
        # "HAR-MV": ["OV", "TV"],
    }
    horizons = [1, 5, 21]
    mse = dict()
    data = get_data("data_files/Todorov-Zhang-JAE-2021.csv")

    for horizon in horizons:
        print(horizon)
        for model, predictors in lr_predictors.items():
            print(model)
            mse[model] = regress(data[predictors], data["RV"], horizon)[0]

        print(mse["HAR-RV"] / mse["HAR-TV"])


if __name__ == "__main__":
    # print(transform_volatility_by_horizon(pd.Series([1, 2, 3, 4, 5, 6, 7]), 5))
    main()
