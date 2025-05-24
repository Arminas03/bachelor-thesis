import pandas as pd
import json
import matplotlib.pyplot as plt

from replication_scripts.replication_regression import get_regression_results
from utils import get_data


def get_regressions_on_rv_args():
    lr_predictors = {
        "HAR-RV": ["RV"],
        "HAR-TV": ["TV"],
        "HAR-OV": ["OV"],
        "HAR-EV": ["EV"],
        "HAR-MV": ["OV", "TV"],
    }
    target = "RV"
    horizons = [1, 5, 22]
    estimation_methods = ["rw", "iw"]

    return lr_predictors, target, horizons, estimation_methods


def get_regressions_on_orv_args():
    lr_predictors = {
        "HAR-TV": ["TV"],
        "HAR-MV": ["OV", "TV"],
        "HAR-EV": ["EV"],
    }
    target = "ORV"
    horizons = [1, 5, 22]
    estimation_methods = ["rw", "iw"]

    return lr_predictors, target, horizons, estimation_methods


def get_regressions_with_jv_args():
    lr_predictors = {
        "HAR-TV": ["TV"],
        "HAR-MV": ["OV", "TV"],
        "HAR-MV-JV": ["OV", "TV", "JV"],
    }
    target = "RV"
    horizons = [1, 5, 22]
    estimation_methods = ["rw", "iw"]

    return lr_predictors, target, horizons, estimation_methods


def reformat_dict(loss_dict):
    return {",".join(map(str, key)): value for key, value in loss_dict.items()}


def plot_estimators_time_series(estimators: pd.DataFrame):
    estimators.index = estimators["date"]
    estimators = estimators.drop("date", axis=1)

    for var_est in estimators.columns:
        estimators[var_est].plot(linewidth=0.6)
        plt.title(f"{var_est}")
        plt.xlabel("Date")
        plt.ylabel(var_est)
        plt.grid(True)
        plt.show()


def get_replication_res_json():
    data = get_data()

    plot_estimators_time_series(data.copy())

    with open("regressions_on_rv_results.json", "w") as f:
        loss_values_on_rv = get_regression_results(*get_regressions_on_rv_args(), data)
        json.dump(reformat_dict(loss_values_on_rv), f)

    with open("regressions_on_orv_results.json", "w") as f:
        loss_values_on_orv = get_regression_results(
            *get_regressions_on_orv_args(), data
        )
        json.dump(reformat_dict(loss_values_on_orv), f)

    with open("regressions_with_jv_results.json", "w") as f:
        loss_values_with_jv = get_regression_results(
            *get_regressions_with_jv_args(), data
        )
        json.dump(reformat_dict(loss_values_with_jv), f)


if __name__ == "__main__":
    get_replication_res_json()
    plot_estimators_time_series(get_data())
