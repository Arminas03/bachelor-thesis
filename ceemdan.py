from PyEMD import CEEMDAN
from utils import get_data
import pandas as pd
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


def main():
    estimators = ["RV", "OV", "TV", "EV", "JV"]
    ceemdan = CEEMDAN(seed=0)
    data = get_data()

    for estimator in estimators:
        decompose_series_with_ceemdan(data[estimator], ceemdan, estimator)

        # print(get_imf_counter("final_imfs_rw_rv.h5"))


if __name__ == "__main__":
    main()
