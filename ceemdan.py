from PyEMD import CEEMDAN
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from utils import get_jae_data
from paths import *


def get_imf_counter(h5_path: str) -> dict:
    """
    Gets IMF counter throughout the period,
    Table 5 in the paper
    """
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
    series: pd.Series, ceemdan: CEEMDAN, estimator_name: str, window_size: int = 1000
) -> None:
    """
    Performs CEEMDAN decomposition
    """
    with h5py.File(IMF_DATA_PATHS[estimator_name], "w") as f_h5:
        for i in range(len(series) - window_size + 1):
            curr_window = series[i : window_size + i].to_numpy()

            f_h5.create_dataset(
                name=f"window_{i}",
                data=ceemdan.ceemdan(curr_window),
                compression="gzip",
            )


def plot_imfs(estimator: str, dates: pd.Series) -> None:
    """
    Plots the resulting IMFs of the first window decomposition
    for a given estimator
    """
    dates = dates[:1000]
    with h5py.File(IMF_DATA_PATHS[estimator], "r") as f_h5:
        imfs = f_h5["window_0"][:]
        n_imfs = imfs.shape[0]

        figure, axes = plt.subplots(n_imfs, 1, figsize=(12, 2 * n_imfs), sharex=True)
        for i in range(n_imfs):
            axes[i].plot(dates, imfs[i], label=f"IMF {i+1}", linewidth=0.6)
            axes[i].legend(loc="upper right")
            axes[i].grid(True)

        axes[-1].set_xlabel("Time")
        figure.canvas.manager.set_window_title(f"IMFs of {estimator}")
        plt.tight_layout()
        plt.show()


def plot_reconstructed_first_window(estimator: str, dates: pd.Series) -> None:
    """
    Plots reconstruction from IMFs of the first decomposed window
    """
    dates = dates[:1000]
    with h5py.File(IMF_DATA_PATHS[estimator], "r") as f_h5:
        imfs = f_h5["window_0"][:]
        reconstructed_estimator = imfs.sum(axis=0)
    estimator_series = get_jae_data()[estimator][0:1000]

    figure = plt.figure(figsize=(12, 6))

    plt.plot(
        dates,
        reconstructed_estimator,
        label=f"Reconstructed {estimator}",
        linewidth=0.8,
        color="blue",
        linestyle="--",
    )
    plt.plot(
        dates,
        estimator_series,
        label=f"True {estimator}",
        linewidth=0.8,
        color="red",
        linestyle=":",
    )

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("RV", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=16)
    figure.canvas.manager.set_window_title(f"Reconstructed vs true of {estimator}")
    plt.tight_layout()
    plt.show()


def ceemdan() -> None:
    """
    Decomposes every estimator with CEEMDAN, saves and plots the results
    """
    estimators = ["RV", "ORV", "OV", "TV", "EV", "JV"]
    ceemdan = CEEMDAN(seed=0)
    data = get_jae_data()

    for estimator in estimators:
        decompose_series_with_ceemdan(data[estimator], ceemdan, estimator)

    for estimator in estimators:
        plot_imfs(estimator, data["date"])
        plot_reconstructed_first_window(estimator, data["date"])


if __name__ == "__main__":
    ceemdan()
