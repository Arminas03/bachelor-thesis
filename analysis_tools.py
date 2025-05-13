import numpy as np
import matplotlib.pyplot as plt


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
