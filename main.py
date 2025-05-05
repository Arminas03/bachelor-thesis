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


def transform_predictors_to_dwm(predictors):
    final_predictors = []
    for _, predictor in predictors.items():
        for horizon in [1, 5, 22]:
            final_predictors.append(
                transform_volatility_by_horizon(predictor, horizon)[(22 - horizon) : -1]
            )

    return final_predictors


def increasing_window_estimation(X, y):
    X_train = X[:-1758]
    y_train, y_test = y[:-1758], y[-1758:]
    y_hat = []
    print(y_test[0])
    print(len(X_train))

    for i in range(1758, -1, -1):
        y_hat.append(
            get_regression_model(X_train, y_train).predict(X[-i].reshape(1, -1))
        )
        X_train = np.append(X_train, X[-i].reshape(1, -1), axis=0)
        y_train = np.append(y_train, [y[-i]], axis=0)

    get_prediction_analysis(y_test, y_hat)


def regress(predictors, true_volatility, horizon):
    predictors_transformed = np.array(transform_predictors_to_dwm(predictors)).T
    true_volatility_transformed = np.array(
        transform_volatility_by_horizon(true_volatility, horizon)[22:]
    )

    increasing_window_estimation(predictors_transformed, true_volatility_transformed)


def main():
    lr_predictors = {"HAR-RV": ["RV"], "HAR-TV": ["TV"]}
    data = get_data("data_files/Todorov-Zhang-JAE-2021.csv")

    regress(data[lr_predictors["HAR-RV"]], data["RV"], 1)
    regress(data[lr_predictors["HAR-TV"]], data["RV"], 1)


if __name__ == "__main__":
    # print(transform_volatility_by_horizon(pd.Series([1, 2, 3, 4, 5, 6, 7]), 5))
    main()
