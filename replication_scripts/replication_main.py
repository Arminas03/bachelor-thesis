import pandas as pd
import json
import matplotlib.pyplot as plt

from replication_scripts.replication_regression import get_regression_results
from utils import get_jae_data


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
    n_estimators = estimators.shape[1]

    figure, axes = plt.subplots(
        n_estimators, 1, figsize=(12, 5 * n_estimators + 2), sharex=True
    )
    for i in range(n_estimators):
        axes[i].plot(
            estimators.iloc[:, i],
            label=f"{estimators.columns[i]}",
            linewidth=0.7,
            color=(
                (0.2 + 0.6 * (i / n_estimators)) % 1,
                (0.5 + 0.4 * (i / n_estimators)) % 1,
                (0.8 - 0.6 * (i / n_estimators)) % 1,
            ),
        )
        axes[i].legend(loc="upper right", fontsize=16)
        axes[i].grid(True)

    for label in axes[-1].get_xticklabels():
        label.set_fontsize(12)

    figure.canvas.manager.set_window_title(f"estimators_time_series")
    plt.tight_layout(pad=7.0)
    plt.show()


def get_replication_res_json():
    data = get_jae_data()

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
    # get_replication_res_json()
    with open("regressions_with_jv_results.json", "w") as f:
        loss_values_with_jv = get_regression_results(
            *get_regressions_with_jv_args(), get_jae_data()
        )
        json.dump(reformat_dict(loss_values_with_jv), f)
