from PyEMD import CEEMDAN
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from utils import get_jae_data
from constants import *


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


def decompose_series_with_ceemdan(
    series: pd.Series, ceemdan: CEEMDAN, estimator_name, window_size=1000, rolling=True
):
    with h5py.File(
        f"final_imfs_{'rw' if rolling else 'iw'}_{estimator_name}.h5", "w"
    ) as f_h5:
        for i in range(len(series) - window_size + 1):
            print("rw" if rolling else "iw", i)
            curr_window = series[i if rolling else 0 : window_size + i].to_numpy()

            f_h5.create_dataset(
                name=f"window_{i}",
                data=ceemdan.ceemdan(curr_window),
                compression="gzip",
            )


def plot_first_imf(estimator):
    with h5py.File(IMF_DATA_PATHS[estimator], "r") as f_h5:
        imfs = f_h5["window_0"][:]
        n_imfs = imfs.shape[0]

        figure, axes = plt.subplots(n_imfs, 1, figsize=(12, 2 * n_imfs), sharex=True)
        for i in range(n_imfs):
            axes[i].plot(imfs[i], label=f"IMF {i+1}", linewidth=0.6)
            axes[i].legend(loc="upper right")
            axes[i].grid(True)

        axes[-1].set_xlabel("Time")
        figure.canvas.manager.set_window_title(f"IMFs of {estimator}")
        plt.tight_layout()
        plt.show()


def main():
    estimators = ["RV", "OV", "TV", "EV", "JV"]
    ceemdan = CEEMDAN(seed=0)
    data = get_jae_data()

    for estimator in estimators:
        decompose_series_with_ceemdan(data[estimator], ceemdan, estimator)


if __name__ == "__main__":
    main()
