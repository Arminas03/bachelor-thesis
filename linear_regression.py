from sklearn.linear_model import LinearRegression
from analysis_tools import *


def get_regression_model(X, y, fit_intercept=True):
    lr_model = LinearRegression(fit_intercept=fit_intercept)
    lr_model.fit(X, y)

    return lr_model


def get_prediction_analysis(y, y_hat):
    plot_true_vs_pred(y_true=y, y_pred=y_hat)
    print(f"MSE: {get_mse(y_true=y, y_pred=y_hat)}")
    print(f"QLIKE: {get_qlike(y_true=y, y_pred=y_hat)}")
