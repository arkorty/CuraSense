import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    # Fill missing values and convert categorical variables to numerical
    data.fillna(0, inplace=True)
    data["Condition"] = data["Condition"].map(
        {"Stable": 0, "Critical": 1, "Life-threatening": 2}
    )
    return data
