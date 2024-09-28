from sklearn.model_selection import train_test_split


def split_data(data):
    X = data.drop(columns=["Condition", "Tentative Recovery Time", "Next Steps"])
    y = data["Condition"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
