from replication_scripts.replication_main import get_replication_res_json
from replication_scripts.return_predictability import get_return_predictability_json
from har_ceemdan_ar import get_har_ceemdan_ar_res_json
from ceemdan import ceemdan, plot_first_imf, plot_reconstructed_first_window
from utils import get_estimator_variance
from mincer_zarnowitz import get_mz_json
from diebold_mariano import get_dm_test_results_for_jv


def main():
    ceemdan()
    get_replication_res_json()
    get_har_ceemdan_ar_res_json()
    get_return_predictability_json()
    get_mz_json()
    get_dm_test_results_for_jv()

    for estimator in ["RV", "ORV", "OV", "TV", "EV", "JV"]:
        plot_first_imf(estimator)
        plot_reconstructed_first_window(estimator)

    for estimator in ["RV", "ORV"]:
        for horizon in [1, 5, 22]:
            print(
                f"Variance of {estimator} at horizon {horizon}: {get_estimator_variance(estimator, horizon)}"
            )


if __name__ == "__main__":
    main()
