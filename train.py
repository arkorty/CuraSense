import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(sys.argv[1])
data = data.drop(columns=["PatientID"])

# Separate features and target labels
X = data.drop(columns=["VentilatorPressure"]).values
y = data["VentilatorPressure"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize/Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data to fit the Transformer model
X_train_scaled = X_train_scaled[..., np.newaxis]  # Add an extra dimension
X_test_scaled = X_test_scaled[..., np.newaxis]


# Define Transformer model
def create_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Transformer Encoder Layer
    x = tf.keras.layers.Conv1D(128, kernel_size=1, activation="relu")(
        inputs
    )  # Initial Conv Layer
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Dropout Layer
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# Create and compile the model
transformer_model = create_transformer_model((X_train_scaled.shape[1], 1))
transformer_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Add Early Stopping to stop training when validation loss doesn't improve
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

# Train the model
history = transformer_model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=300,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluate the model
test_loss, test_mae = transformer_model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the model
transformer_model.save("ventilator_adjustment_transformer_model.keras")
