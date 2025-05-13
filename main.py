from regression_results import get_regression_results
import json


def get_regressions_on_rv_args():
    lr_predictors = {
        "HAR-RV": ["RV"],
        "HAR-TV": ["TV"],
        "HAR-OV": ["OV"],
        "HAR-EV": ["EV"],
        "HAR-MV": ["OV", "TV"],
    }
    target = "RV"
    horizons = [1, 5, 22]
    estimation_methods = ["rw", "iw"]

    return lr_predictors, target, horizons, estimation_methods


def get_regressions_on_orv_args():
    lr_predictors = {
        "HAR-TV": ["TV"],
        "HAR-MV": ["OV", "TV"],
        "HAR-EV": ["EV"],
    }
    target = "ORV"
    horizons = [1, 5, 22]
    estimation_methods = ["rw", "iw"]

    return lr_predictors, target, horizons, estimation_methods


def get_regressions_with_jv_args():
    lr_predictors = {
        "HAR-TV": ["TV"],
        "HAR-MV": ["OV", "TV"],
        "HAR-MV-JV": ["OV", "TV", "JV"],
    }
    target = "RV"
    horizons = [1, 5, 22]
    estimation_methods = ["rw", "iw"]

    return lr_predictors, target, horizons, estimation_methods


def main():
    with open("regressions_on_rv_results.json", "w") as f:
        json.dump(get_regression_results(**get_regressions_on_rv_args), f)

    with open("regressions_on_orv_results.json", "w") as f:
        json.dump(get_regression_results(**get_regressions_on_orv_args), f)

    with open("regressions_with_jv_results.json", "w") as f:
        json.dump(get_regression_results(**get_regressions_with_jv_args), f)


if __name__ == "__main__":
    main()
