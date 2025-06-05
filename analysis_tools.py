import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import get_jae_data, transform_volatility_by_horizon


def plot_true_vs_pred(y_true, y_pred):
    plt.plot(y_true, color="blue", label="True volatility")
    plt.plot(y_pred, color="red", label="Predicted volatility")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def get_squared_errors(y_true, y_pred):
    squared_errors = (y_true - y_pred) ** 2
    return float(np.mean(squared_errors)), float(np.median(squared_errors))


def get_qlike(y_true, y_pred):
    qlike = y_true / y_pred - np.log(y_true / y_pred) - 1
    return float(np.mean(qlike)), float(np.median(qlike))


def get_estimator_variance(estimator, horizon):
    series = transform_volatility_by_horizon(get_jae_data()[estimator][1000:], horizon)
    return np.var(np.array(series))


if __name__ == "__main__":
    for estimator in ["RV", "ORV"]:
        print("RV:")
        for horizon in [1, 5, 22]:
            print(horizon)
            print(get_estimator_variance(estimator, horizon))
