import pandas as pd
import statsmodels.api as sm
import numpy as np
import json
import matplotlib.pyplot as plt

from paths import *
from utils import get_jae_data, transform_volatility_by_horizon


def get_vix_close_data(from_: str = "2008-01-01", to: str = "2019-01-01") -> pd.Series:
    """
    Gets VIX close data for given dates
    """
    vix_data = pd.read_csv(PATH_TO_VIX_DATA, index_col=0)["CLOSE"]

    vix_data.index = pd.to_datetime(vix_data.index)

    return vix_data.loc[(vix_data.index > from_) & (vix_data.index < to)].rename("VIX")


def get_mkt_excess_return_data(
    from_: str = "2008-01-01", to: str = "2019-01-01"
) -> pd.Series:
    """
    Gets Kenneth R. French excess market return data for given dates
    """
    mkt_ret = pd.read_csv(
        PATH_TO_MKT_RF_RET_DATA, index_col=0, skiprows=3, skipfooter=1
    )["Mkt-RF"]

    mkt_ret.index = pd.to_datetime(mkt_ret.index, format="%Y%m%d")

    return mkt_ret.loc[(mkt_ret.index > from_) & (mkt_ret.index < to)]


def get_indexed_jae_data() -> pd.DataFrame:
    """
    Gets JAE data and indexes it according to date
    """
    data_jae = get_jae_data()
    data_jae.index = data_jae["date"]

    return data_jae


def trim_series(series: pd.Series, allowed_dates: set) -> np.array:
    """
    Returns series with only allowed dates. Used for inner joining
    excess return and VIX close data with Todorov and Zhang (2022) data
    """
    return series.loc[series.index.isin(allowed_dates)].to_numpy()


def get_returns_over_horizon(ret: np.array, h: int) -> np.array:
    """
    Compounds returns over given horizon
    """
    ret_over_h = np.empty(len(ret) - h)

    for i in range(len(ret) - h):
        curr_ret = 1
        for j in range(i, i + h + 1):
            curr_ret *= 1 + ret[j]
        ret_over_h[i] = curr_ret - 1

    return ret_over_h


def get_var_series(
    jae_data: pd.DataFrame, horizon: int
) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Gets series for each variable transformed for monthly volatilities,
    for given horizon
    """
    end = -horizon
    return (
        np.array(transform_volatility_by_horizon(jae_data["RV"], 22))[:end],
        np.array(transform_volatility_by_horizon(jae_data["ORV"], 22))[:end],
        np.array(transform_volatility_by_horizon(jae_data["JV"], 22))[:end],
        np.array(transform_volatility_by_horizon(jae_data["TV"], 22))[:end],
    )


def get_vrp(vix_series: np.array, rv_series: np.array) -> np.array:
    """
    Returns VRP for given series
    """
    return (vix_series / 100) ** 2 - rv_series


def get_jrp(jv_series: np.array, rv_series: np.array, tv_series: np.array) -> np.array:
    """
    Returns JRP
    """
    return jv_series - (rv_series - tv_series)


def get_ols_statistics(X: np.array, y: np.array) -> tuple[float, float]:
    """
    Gets t-values and R-squared for OLS. Note that
    t-values are not used in the research.
    """
    model = sm.OLS(y, sm.add_constant(X)).fit()

    return model.tvalues[1], model.rsquared


def get_regression_statistics_for_regressor(regressor: np.array, mkt: np.array) -> dict:
    """
    Returns dict of resulting t-statistics and r-squared
    """
    t_statistic, r_squared = get_ols_statistics(regressor, mkt)

    return {"t_statistic": t_statistic, "r_squared": r_squared}


def get_regression_statistics(
    mkt: np.array, vrp_1: np.array, vrp_2: np.array, jrp: np.array
) -> dict:
    """
    Returns result dict for VRP1, VRP2, and JRP regressions
    """
    res_dict_for_horizon = dict()

    res_dict_for_horizon["vrp_1"] = get_regression_statistics_for_regressor(vrp_1, mkt)
    res_dict_for_horizon["vrp_2"] = get_regression_statistics_for_regressor(vrp_2, mkt)
    res_dict_for_horizon["jrp"] = get_regression_statistics_for_regressor(jrp, mkt)

    return res_dict_for_horizon


def plot_predictability_rsquared() -> None:
    """
    Plots R-squared time series for given return horizon
    """
    with open("return_predictability.json", "r") as f:
        data = json.load(f)

    horizons = range(1, 13)
    vrp_1_r2 = [data[f"horizon_{h}_month"]["vrp_1"]["r_squared"] for h in horizons]
    vrp_2_r2 = [data[f"horizon_{h}_month"]["vrp_2"]["r_squared"] for h in horizons]
    jrp_r2 = [data[f"horizon_{h}_month"]["jrp"]["r_squared"] for h in horizons]

    figure = plt.figure(figsize=(10, 5))
    plt.plot(horizons, vrp_1_r2, marker="x", linestyle="None", label="VRP1")
    # fmt: off
    plt.plot(
        horizons, vrp_2_r2,marker="o",linestyle="None",label="VRP2", markerfacecolor="none"
    )
    # fmt: on
    plt.plot(horizons, jrp_r2, marker="$*$", linestyle="None", label="JRP")

    figure.canvas.manager.set_window_title("return_predictability_R2_plot")
    plt.xlabel("Horizon in months", fontsize=14)
    plt.ylabel("R2", fontsize=18)
    plt.xticks(horizons)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_return_predictability_json() -> None:
    """
    Saves .json of return predictability regressions
    """
    jae_data = get_indexed_jae_data()
    allowed_dates = set(jae_data["date"])
    vix_series = trim_series(get_vix_close_data(), allowed_dates)
    mkt_ret_series = trim_series(get_mkt_excess_return_data(), allowed_dates)

    res_dict = dict()

    for month in range(1, 13):
        horizon = month * 22
        curr_vix = vix_series[21:-horizon]
        curr_mkt_ret = get_returns_over_horizon((mkt_ret_series / 100)[21:], horizon)

        rv_series, orv_series, jv_series, tv_series = get_var_series(jae_data, horizon)

        res_dict[f"horizon_{month}_month"] = get_regression_statistics(
            curr_mkt_ret,
            get_vrp(curr_vix, rv_series),
            get_vrp(curr_vix, orv_series),
            get_jrp(jv_series, rv_series, tv_series),
        )

    with open("return_predictability.json", "w") as f:
        json.dump(res_dict, f)

    plot_predictability_rsquared()


if __name__ == "__main__":
    get_return_predictability_json()
