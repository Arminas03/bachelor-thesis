import numpy as np
from scipy.stats import norm
import json

from regression_common import rolling_window, increasing_window, regress
from utils import get_jae_data


def get_mse_arr(y_true, y_pred):
    return (y_true - y_pred) ** 2


def get_qlike_arr(y_true, y_pred):
    return y_true / y_pred - np.log(y_true / y_pred) - 1


def get_dm_p_value(d_arr):
    dm_statistic = np.sqrt(len(d_arr)) * np.mean(d_arr) / np.std(d_arr, ddof=1)
    return {"statistic": dm_statistic, "p_value": 1 - norm.cdf(abs(dm_statistic))}


def get_dm_analysis(y_true, y_pred_1, y_pred_2):
    d_mse = (
        get_mse_arr(y_true, y_pred_1)
        - get_mse_arr(y_true, y_pred_2)
        + (y_pred_1 - y_pred_2) ** 2
    )

    return get_dm_p_value(d_mse)


def get_dm_result_dict(
    predictors_model_1, predictors_model_2, target, horizons, estimation_methods, data
):
    # Model 1 must be nested in model 2
    results = dict()
    for horizon in horizons:
        print(horizon)
        results[horizon] = dict()
        for estimation_method in estimation_methods:
            print(estimation_method)
            estimation_function = (
                rolling_window if estimation_method == "rw" else increasing_window
            )

            y_true, y_pred_model_1 = regress(
                data[predictors_model_1], data[target], horizon, estimation_function
            )
            _, y_pred_model_2 = regress(
                data[predictors_model_2], data[target], horizon, estimation_function
            )

            results[horizon][estimation_method] = get_dm_analysis(
                np.array(y_true), np.array(y_pred_model_1), np.array(y_pred_model_2)
            )

    return results


def get_dm_test_results_for_jv():
    dm_result_dict = get_dm_result_dict(
        ["OV", "TV"],  # HAR-MV
        ["OV", "TV", "JV"],  # HAR-MV-JV
        "RV",
        [1, 5, 22],
        ["rw", "iw"],
        get_jae_data(),
    )

    with open("jv_dm_test.json", "w") as f:
        json.dump(dm_result_dict, f)


if __name__ == "__main__":
    get_dm_test_results_for_jv()
