import pandas as pd
from data_processing import load_data, preprocess_data
from feature_engineering import split_data
from model import create_model
from evaluation import evaluate_model


def train_model():
    # Load and preprocess the data
    data = load_data("data/dummy_data.csv")
    data = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Create the model
    model = create_model((X_train.shape[1],))

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    train_model()
