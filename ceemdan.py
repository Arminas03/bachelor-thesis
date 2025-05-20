from PyEMD import CEEMDAN
from utils import get_data
import time
from linear_regression import get_regression_model
import numpy as np
import pandas as pd
from regression_results import (
    transform_predictors_to_dwm,
    transform_volatility_by_horizon,
)
import h5py


def get_imf_counter(h5_path):
    with h5py.File(h5_path, "r") as f_h5:
        imf_counter = dict()
        for key in list(f_h5.keys()):
            num_imf = len(f_h5[key])
            if num_imf in imf_counter:
                imf_counter[num_imf] += 1
            else:
                imf_counter[num_imf] = 1

    return imf_counter


def test():
    rv_ts = get_data()["RV"]
    y_pred_har = []
    y_pred_imf = []
    y_actual = []
    x_train = np.array(transform_predictors_to_dwm(pd.DataFrame(rv_ts), 1)).T
    y_train = np.array(transform_volatility_by_horizon(pd.Series(rv_ts), 1)[22:])

    with h5py.File("final_imfs_rw_rv.h5", "r") as f_h5:
        for i in range(len(rv_ts)):
            curr_imf = f_h5[f"window_{i+23}"][:]

            x_train_window = x_train[i : 999 + i]
            y_train_window = y_train[i : 999 + i]

            curr_imf = curr_imf.T

            y_pred_har.append(
                get_regression_model(x_train_window, y_train_window).predict(
                    x_train[999 + i].reshape(1, -1)
                )[0]
            )
            y_pred_imf.append(
                get_regression_model(curr_imf[:-1], y_train_window).predict(
                    curr_imf[-1].reshape(1, -1)
                )[0]
            )

            y_actual.append(y_train[999 + i])
            if i == 1700:
                break

        y_actual = np.array(y_actual)
        y_pred_imf = np.array(y_pred_imf)
        y_pred_har = np.array(y_pred_har)

        print(np.mean((y_actual - y_pred_har) ** 2))
        print(np.mean((y_actual - y_pred_imf) ** 2))


def decompose_series_with_ceemdan(
    series: pd.Series, ceemdan: CEEMDAN, window_size=1000, rolling=True
):
    # TODO: change name
    with h5py.File(f"final_imfs_{'rw' if rolling else 'iw'}.h5", "a") as f_h5:
        for i in range(len(series) - window_size + 1):
            print("rw" if rolling else "iw", i)
            curr_window = series[i if rolling else 0 : window_size + i].to_numpy()

            f_h5.create_dataset(
                name=f"window_{i}",
                data=ceemdan.ceemdan(curr_window),
                compression="gzip",
            )


def main():
    # TODO: finish
    estimators = ["RV", "OV", "TV", "EV", "JV"]
    ceemdan = CEEMDAN(seed=0)

    for estimator in estimators:
        decompose_series_with_ceemdan(get_data()[estimator], ceemdan)


if __name__ == "__main__":
    # ceemdan = CEEMDAN(seed=0)
    # data = get_data()["EV"]

    # decompose_series_with_ceemdan(data, ceemdan)
    print(get_imf_counter("final_imfs_rw_ev.h5"))
