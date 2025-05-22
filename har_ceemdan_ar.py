import numpy as np
import h5py


from utils import get_data
from regression_common import (
    regress,
    rolling_window,
    get_prediction_loss,
    get_regression_model,
    winsorise,
)
from analysis_tools import plot_true_vs_pred


def get_har_pred(target, pred_var):
    data = get_data()
    target_ts = data[target]
    predictor_ts = data[[pred_var]]

    return regress(predictor_ts, target_ts, horizon=1, estimation_method=rolling_window)


def get_ceemdan_ar_pred(estimator_name, window_size=1000):
    # 22 first observations must be discarded due to comparison with HAR-RV
    # Pushed by 1 due to future
    rv_ts = np.array(get_data()["RV"][21 + 1 :])
    y_pred_imf = []
    y_test = []

    with h5py.File(
        f"final_imfs/final_imfs_rw_{estimator_name.lower()}.h5", "r"
    ) as f_h5:
        for i in range(len(rv_ts) - window_size):
            curr_imf = f_h5[f"window_{i+21}"][:].T
            y_train_window = rv_ts[i : window_size + i - 1]

            y_test.append(rv_ts[window_size + i - 1])

            y_pred = get_regression_model(curr_imf[:-1], y_train_window).predict(
                curr_imf[-1].reshape(1, -1)
            )[0]
            y_pred_imf.append(winsorise(y_train_window, y_pred))

    return np.array(y_test), np.array(y_pred_imf)


def main():
    y_har_true, y_har_pred = get_har_pred("RV", "RV")
    print(f"Loss HAR-RV: {get_prediction_loss(y_har_true, y_har_pred)}")
    y_car_true, y_car_pred = get_ceemdan_ar_pred("RV")
    print(f"Loss CEEMDAN-AR: {get_prediction_loss(y_car_true, y_car_pred)}")
    plot_true_vs_pred(y_har_true, y_har_pred)
    plot_true_vs_pred(y_car_true, y_car_pred)


if __name__ == "__main__":
    main()
