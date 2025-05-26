from regression_common import (
    rolling_window,
    increasing_window,
    regress,
    get_prediction_loss,
)


def assign_loss_values(
    loss_values, regression_results, horizon, estimation_method, model
):
    for loss_name, agg_loss_values in [
        ("squared_error", regression_results[0]),
        ("qlike", regression_results[1]),
    ]:
        loss_values[(horizon, estimation_method, model, loss_name, "mean")] = (
            agg_loss_values[0]
        )
        loss_values[(horizon, estimation_method, model, loss_name, "median")] = (
            agg_loss_values[1]
        )


def standardize_loss(loss_values, by_model):
    # Note: index 2 refers to the model
    for spec, _ in loss_values.items():
        if spec[2] == by_model:
            continue
        loss_values[spec] = (
            loss_values[spec]
            / loss_values[(spec[0], spec[1], by_model, spec[3], spec[4])]
        )
    for spec, _ in loss_values.items():
        if spec[2] == by_model:
            loss_values[spec] = 1


def get_regression_results(lr_predictors, target, horizons, estimation_methods, data):
    loss_values = dict()

    for horizon in horizons:
        for estimation_method in estimation_methods:
            estimation_function = (
                rolling_window if estimation_method == "rw" else increasing_window
            )
            for model, predictors in lr_predictors.items():
                pred_loss = get_prediction_loss(
                    *regress(
                        data[predictors], data[target], horizon, estimation_function
                    )
                )
                assign_loss_values(
                    loss_values, pred_loss, horizon, estimation_method, model
                )

                print(
                    f"Finished results for horizon {horizon}, "
                    + f"estimation method {estimation_method}, model {model}"
                )

    standardize_loss(loss_values, "HAR-TV")

    return loss_values
