import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=input_shape),
            layers.Dense(32, activation="relu"),
            layers.Dense(
                3, activation="softmax"
            ),  # 3 classes: Stable, Critical, Life-threatening
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
