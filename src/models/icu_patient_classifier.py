import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class ICUPatientClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes, learning_rate=0.001):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.label_encoder = LabelEncoder()

    def _build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128, activation="relu", input_shape=(self.input_dim,)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(
            X,
            y_encoded,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        return self

    def predict(self, X):
        y_pred_proba = self.model.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_encoded = self.label_encoder.transform(y)
        return self.model.evaluate(X, y_encoded)


# Usage example
if __name__ == "__main__":
    # Assume we have preprocessed and engineered data
    X_train = np.random.rand(1000, 50)  # 1000 samples, 50 features
    y_train = np.random.choice(["Stable", "Critical", "Emergency"], 1000)

    clf = ICUPatientClassifier(input_dim=50, num_classes=3)
    clf.fit(X_train, y_train)

    X_test = np.random.rand(100, 50)
    y_pred = clf.predict(X_test)

    print("Model training completed.")
    print(f"Predicted classes: {y_pred[:10]}")
