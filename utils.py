import pandas as pd


def get_data(path="data_files/Todorov-Zhang-JAE-2021.csv"):
    data = pd.read_csv(path).rename(columns={"???Date": "date"})
    data["date"] = pd.to_datetime(data["date"])

    return data


def transform_volatility_by_horizon(true_volatility, horizon):
    """
    Transforms volatility by given horizon.
    Note that the last horizon - 1 observations are lost
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


def transform_predictors_to_dwm(predictors, horizon):
    """
    Transforms predictors to daily, weekly, monthly format.
    Note that after the transformation, 21 first valus are lost
    due to inability to calculate monthly predictor value
    """
    final_predictors = []
    for _, predictor in predictors.items():
        for steps in [1, 5, 22]:
            final_predictors.append(
                transform_volatility_by_horizon(predictor, steps)[
                    (22 - steps) : -horizon
                ]
            )

    return final_predictors
