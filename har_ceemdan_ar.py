import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
import json


from utils import get_jae_data, transform_volatility_by_horizon, print_coef_analysis
from regression_common import (
    regress,
    rolling_window,
    get_prediction_loss,
    get_regression_model,
    winsorise,
)
from paths import *


def get_har_pred(target_estimator, predictor_estimators, h):
    data = get_jae_data()
    target_ts = data[target_estimator]
    predictor_ts = data[predictor_estimators]

    return regress(predictor_ts, target_ts, h, rolling_window)


def get_log_har_pred(target_estimator, predictor_estimators, h):
    data = get_jae_data()
    target_ts = data[target_estimator]
    predictor_ts = data[predictor_estimators]

    y_true, y_pred = regress(predictor_ts, target_ts, h, rolling_window, True)

    return np.exp(y_true), np.exp(y_pred)


def get_curr_imfs(i, predictor_estimators):
    imfs = []
    for predictor_estimator in predictor_estimators:
        with h5py.File(IMF_DATA_PATHS[predictor_estimator], "r") as f_h5:
            imfs.append(f_h5[f"window_{i+21}"][:].T)

    return np.concatenate(imfs, axis=1)


def get_ceemdan_ar_pred(target_estimator, predictor_estimators, h, window_size=1000):
    # 22 first observations must be discarded due to comparison with HAR-RV
    # Pushed by 1 due to future
    rv_ts = np.array(
        transform_volatility_by_horizon(get_jae_data()[target_estimator], h)[21 + 1 :]
    )
    y_pred_imf = []
    y_test = []

    for i in range(len(rv_ts) - window_size - h + 1):
        curr_imf = get_curr_imfs(i, predictor_estimators)

        y_train_window = rv_ts[i : window_size + i - 1]
        y_test.append(rv_ts[window_size + i - 1])

        imf_scaler, y_scaler = StandardScaler(), StandardScaler()
        curr_imf = imf_scaler.fit_transform(curr_imf)
        y_train_window = y_scaler.fit_transform(y_train_window.reshape(-1, 1)).ravel()

        if i == 0 and target_estimator == "RV":
            print_coef_analysis(curr_imf[:-1], y_train_window)

        model = get_regression_model(curr_imf[:-1], y_train_window)
        y_pred = model.predict(curr_imf[-1].reshape(1, -1))[0]
        y_pred_imf.append(
            winsorise(
                y_scaler.inverse_transform(y_train_window.reshape(-1, 1)).ravel(),
                y_scaler.inverse_transform([[y_pred]])[0, 0],
            )
        )

    return np.array(y_test), np.array(y_pred_imf)


def add_losses_to_dict(losses):
    return {
        "squared_error": {
            "mean": losses[0][0],
            "median": losses[0][1],
        },
        "qlike": {
            "mean": losses[1][0],
            "median": losses[1][1],
        },
    }


def update_res_dict(res_dict, models_regressors, target, horizon):
    for model_name, regressors in models_regressors.items():
        losses_har = get_prediction_loss(*get_har_pred(target, regressors, horizon))
        losses_ceemdan_ar = get_prediction_loss(
            *get_ceemdan_ar_pred(target, regressors, horizon)
        )
        losses_log_har = get_prediction_loss(
            *get_log_har_pred(target, regressors, horizon)
        )

        res_dict[f"HAR-{model_name}"] = add_losses_to_dict(losses_har)
        res_dict[f"CEEMDAN-AR-{model_name}"] = add_losses_to_dict(losses_ceemdan_ar)
        res_dict[f"log-HAR-{model_name}"] = add_losses_to_dict(losses_log_har)

        print(f"Finished {model_name}")


def get_res_dict(models_regressors, target, horizon=1):
    res_dict = dict()

    update_res_dict(res_dict, {f"{target}": [target]}, target, horizon)
    update_res_dict(res_dict, models_regressors, target, horizon)

    return res_dict


def get_har_ceemdan_ar_res_json():
    models_regressors = {
        "MV-JV": ["TV", "OV", "JV"],
        "EV": ["EV"],
    }
    targets = ["RV", "ORV"]

    for target in targets:
        print(target)
        for horizon in [1, 5, 22]:
            print(f"Horizon: {horizon}")
            with open(f"har_ceemdan_ar_results_h_{horizon}_{target}.json", "w") as f:
                json.dump(get_res_dict(models_regressors, target, horizon), f)


if __name__ == "__main__":
    get_har_ceemdan_ar_res_json()
