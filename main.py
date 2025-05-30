from replication_scripts.replication_main import get_replication_res_json
from replication_scripts.return_predictability import get_return_predictability_json
from har_ceemdan_ar import get_har_ceemdan_ar_res_json
from ceemdan import plot_first_imf, plot_reconstructed_first_window


def main():
    get_replication_res_json()
    get_har_ceemdan_ar_res_json()
    get_return_predictability_json()

    for estimator in ["RV", "ORV", "OV", "TV", "EV", "JV"]:
        plot_first_imf(estimator)
        plot_reconstructed_first_window(estimator)


if __name__ == "__main__":
    main()
