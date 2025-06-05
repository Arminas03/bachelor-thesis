import numpy as np
import statsmodels.api as sm
import json

from har_ceemdan_ar import get_log_har_pred


def run_mz(y_true: np.array, y_pred: np.array):
    model = sm.OLS(y_true.reshape(-1), sm.add_constant(y_pred.reshape(-1, 1))).fit()

    return {
        "alpha": float(model.params[0]),
        "beta": float(model.params[1]),
        "R-Squared": float(model.rsquared),
        "alpha_zero_test_p_val": float(model.pvalues[0]),
        "beta_one_test_p_val": float(model.t_test("x1 = 1").pvalue),
    }


def get_mz_res_dict():
    mz_res = dict()
    for estimator in ["RV", "ORV"]:
        mz_res[estimator] = dict()
        for horizon in [1, 5, 22]:
            mz_res[estimator][horizon] = run_mz(
                *get_log_har_pred(estimator, [estimator], horizon)
            )

    return mz_res


def get_mz_json():
    with open("mz_log_har_res.json", "w") as f:
        json.dump(get_mz_res_dict(), f)


if __name__ == "__main__":
    get_mz_json()
