import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT


from paths import *


def get_jae_data() -> pd.DataFrame:
    """
    Extracts and returns Todorov and Zhang (2022) data from JAE data archive
    """
    data = pd.read_csv(PATH_TO_JAE_DATA).rename(columns={"???Date": "date"})
    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%y")

    return data


def print_coef_analysis(X: np.array, y: np.array) -> None:
    """
    Generates results for Table 6, that is,
    prints regression results
    """
    print("\n\n\n")
    print("-" * 50)
    print("First window regression coefficient analysis")
    print("-" * 50)
    X_train = statsmodels.api.add_constant(X)
    model = RLM(y, X_train, M=HuberT()).fit()

    df = pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "z": model.tvalues,
            "p>|z|": model.pvalues,
        }
    )
    df.index = ["const"] + [f"IMF{i}" for i in range(1, 9)]

    print(df.round(6))
    print("\n\n\n")


def transform_volatility_by_horizon(true_volatility: pd.Series, horizon: int) -> list:
    """
    Transforms volatility by given horizon.
    Note that the first horizon - 1 observations are lost
    """
    transformed_vol = []
    c = 1
    curr_sum_vol = 0
    for i, true_vol in enumerate(true_volatility):
        curr_sum_vol += true_vol
        if c < horizon:
            c += 1
            continue

        transformed_vol.append(curr_sum_vol)

        curr_sum_vol -= true_volatility.iloc[i - horizon + 1]

    return transformed_vol


def transform_predictors_to_dwm(predictors: pd.DataFrame, horizon: int) -> list[list]:
    """
    Transforms predictors to daily, weekly, monthly format.
    Note that after the transformation, 21 first values are lost
    due to inability to calculate monthly predictor value
    """
    final_predictors = []
    for name, predictor in predictors.items():
        for steps in [1, 5, 22]:
            final_predictors.append(
                transform_volatility_by_horizon(predictor, steps)[
                    (22 - steps) : -horizon
                ]
            )
            if name == "JV":
                # JV only 1 step back
                break

    return final_predictors


def plot_true_vs_pred(y_true: np.array, y_pred: np.array):
    """
    Plots true vs predicted values. Not used in the main code nor research,
    but good for visually evaluating forecasts
    """
    plt.plot(y_true, color="blue", label="True volatility")
    plt.plot(y_pred, color="red", label="Predicted volatility")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def get_squared_errors(y_true: np.array, y_pred: np.array) -> tuple[float, float]:
    """
    Gets squared error mean and median values
    """
    squared_errors = (y_true - y_pred) ** 2
    return float(np.mean(squared_errors)), float(np.median(squared_errors))


def get_qlike(y_true: np.array, y_pred: np.array) -> tuple[float, float]:
    """
    Gets qlike mean and median values
    """
    qlike = y_true / y_pred - np.log(y_true / y_pred) - 1
    return float(np.mean(qlike)), float(np.median(qlike))


def get_estimator_variance(estimator: str, horizon: int) -> float:
    """
    Returns variance for requested estimator
    """
    series = transform_volatility_by_horizon(get_jae_data()[estimator][1000:], horizon)
    return np.var(np.array(series))
