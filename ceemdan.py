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


def main():
    data = get_data()["RV"]
    y_pred_har = []
    y_pred_ceemdan_har = []
    actual = []
    ceemdan = CEEMDAN(seed=1)

    for i in range(len(data) - 1000):
        rv_ts = data[i : 1000 + i].to_numpy()
        x_train = np.array(transform_predictors_to_dwm(pd.DataFrame(rv_ts), 1)).T

        imfs = ceemdan.ceemdan(rv_ts)
        y_train = np.array(transform_volatility_by_horizon(pd.Series(rv_ts), 1)[22:])
        y_pred_har.append(
            get_regression_model(x_train, y_train).predict(x_train[-1].reshape(1, -1))[
                0
            ]
        )

        y_pred_curr = []

        for imf in imfs:
            x_train = np.array(transform_predictors_to_dwm(pd.DataFrame(imf), 1)).T
            y_train = np.array(transform_volatility_by_horizon(pd.Series(imf), 1)[22:])

            y_pred_curr.append(
                get_regression_model(x_train, y_train).predict(
                    x_train[-1].reshape(1, -1)
                )[0]
            )
        y_pred_ceemdan_har.append(np.sum(y_pred_curr))
        actual.append(data[i + 1000])
        print(i)

        if i == 500:
            break

    print(np.mean((np.array(y_pred_har) - np.array(actual)) ** 2))
    print(np.mean((np.array(y_pred_ceemdan_har) - np.array(actual)) ** 2))


def decompose_series_with_ceemdan(
    series: pd.Series, ceemdan: CEEMDAN, window_size=1000, rolling=True
):
    with h5py.File(f"final_imfs_{'rw' if rolling else 'iw'}.h5", "w") as f_h5:
        for i in range(len(series) - window_size + 1):
            print("rw" if rolling else "iw", i)
            curr_window = series[i : window_size + i].to_numpy()

            f_h5.create_dataset(
                name=f"window_{i}",
                data=ceemdan.ceemdan(curr_window),
                compression="gzip",
            )
            break


if __name__ == "__main__":
    # ceemdan = CEEMDAN(seed=0)
    # data = get_data()["RV"]

    # decompose_series_with_ceemdan(data, ceemdan)
    # decompose_series_with_ceemdan()
    main()
