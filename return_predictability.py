import pandas as pd
import statsmodels.api as sm
import numpy as np

from constants import *
from utils import get_data, transform_volatility_by_horizon


def get_vix_close_data(from_="2008-01-01", to="2019-01-01"):
    vix_data = pd.read_csv(PATH_TO_VIX_DATA, index_col=0)["CLOSE"]

    vix_data.index = pd.to_datetime(vix_data.index)

    return vix_data.loc[(vix_data.index > from_) & (vix_data.index < to)].rename("VIX")


def get_mkt_excess_return_data(from_="2008-01-01", to="2019-01-01"):
    mkt_ret = pd.read_csv(
        PATH_TO_MKT_RF_RET_DATA, index_col=0, skiprows=3, skipfooter=1
    )["Mkt-RF"]

    mkt_ret.index = pd.to_datetime(mkt_ret.index, format="%Y%m%d")

    return mkt_ret.loc[(mkt_ret.index > from_) & (mkt_ret.index < to)]


def get_jae_data():
    data_jae = get_data()
    data_jae.index = data_jae["date"]

    return data_jae


def trim_series(series: pd.Series, allowed_dates):
    return series.loc[series.index.isin(allowed_dates)]


def get_var_series(jae_data):
    return (
        np.array(transform_volatility_by_horizon(jae_data["RV"], 22)),
        np.array(transform_volatility_by_horizon(jae_data["ORV"], 22)),
        np.array(transform_volatility_by_horizon(jae_data["JV"], 22)),
        np.array(transform_volatility_by_horizon(jae_data["TV"], 22)),
    )


def get_vrp(vix_series, rv_series):
    return ((vix_series / 100) ** 2 - rv_series).to_numpy()


def get_jrp(jv_series, rv_series, tv_series):
    return jv_series - (rv_series - tv_series)


def get_ols_summary(x, y):
    return sm.OLS(y, sm.add_constant(x)).fit().summary()


def run_regressions(mkt, vrp_1, vrp_2, jrp):
    print(get_ols_summary(jrp[:-1], mkt[1:]))


def main():
    vix_series = get_vix_close_data()
    mkt_ret_series = get_mkt_excess_return_data()
    jae_data = get_jae_data()

    allowed_dates = set(jae_data["date"])
    vix_series = trim_series(vix_series, allowed_dates)[21:]
    mkt_ret_series = trim_series(mkt_ret_series, allowed_dates)
    mkt_ret_series = transform_volatility_by_horizon(mkt_ret_series, 22)

    rv_series, orv_series, jv_series, tv_series = get_var_series(jae_data)

    run_regressions(
        mkt_ret_series,
        get_vrp(vix_series, rv_series),
        get_vrp(vix_series, orv_series),
        get_jrp(jv_series, rv_series, tv_series),
    )


if __name__ == "__main__":
    main()
