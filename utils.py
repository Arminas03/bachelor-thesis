import pandas as pd


def get_data(path="data_files/Todorov-Zhang-JAE-2021.csv"):
    data = pd.read_csv(path).rename(columns={"???Date": "date"})
    data["date"] = pd.to_datetime(data["date"])

    return data
