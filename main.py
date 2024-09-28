import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.icu_patient_classifier import ICUPatientClassifier
from src.utils.evaluation_metrics import calculate_metrics


def main():
    # Load data
    sensor_data = pd.read_csv("data/raw/sensor_data.csv")
    admission_data = pd.read_csv("data/raw/admission_data.csv")

    # Preprocess data
    preprocessor = DataPreprocessor()
    sensor_data_processed = preprocessor.preprocess_sensor_data(sensor_data)
    admission_data_processed = preprocessor.preprocess_admission_data(admission_data)
    merged_data = preprocessor.merge_data(
        sensor_data_processed, admission_data_processed
    )

    # Feature engineering
    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.transform(merged_data)

    # Prepare data for modeling
    X = engineered_data.drop(columns=["patient_id", "severity"])
    y = engineered_data["severity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    clf = ICUPatientClassifier(input_dim=X_train.shape[1], num_classes=3)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Evaluate model
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Example of making predictions for new patients
    new_patient_data = X_test.iloc[:5]  # Use first 5 samples from test set as example
    new_patient_predictions = clf.predict(new_patient_data)
    new_patient_probabilities = clf.predict_proba(new_patient_data)

    print("\nPredictions for new patients:")
    for i, (pred, probs) in enumerate(
        zip(new_patient_predictions, new_patient_probabilities)
    ):
        print(f"Patient {i+1}:")
        print(f"  Predicted condition: {pred}")
        print(
            f"  Probabilities: Stable: {probs[0]:.2f}, Critical: {probs[1]:.2f}, Emergency: {probs[2]:.2f}"
        )

        if pred == "Stable":
            recovery_time = np.random.randint(1, 10)  # Simplified estimation
            print(f"  Estimated recovery time: {recovery_time} days")
        elif pred == "Critical":
            print("  Recommended steps for next 24 hours:")
            print("    - Monitor vital signs closely")
            print("    - Adjust medication as needed")
            print("    - Prepare for potential interventions")
        else:  # Emergency
            print("  SOS Alert: Immediate medical intervention required")
            print("  Critical updates:")
            print("    - Prepare emergency response team")
            print("    - Notify on-call specialists")
            print("    - Ready life-support equipment")
        print()


if __name__ == "__main__":
    main()
