import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_pred(y_true, y_pred):
    plt.plot(y_true, color="blue", label="True volatility")
    plt.plot(y_pred, color="red", label="Predicted volatility")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def get_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def get_qlike(y_true, y_pred):
    return float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1))
