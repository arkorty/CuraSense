import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Calculate BMI (handle missing height gracefully)
        if "height" in X_transformed.columns and "body_weight" in X_transformed.columns:
            X_transformed["bmi"] = X_transformed["body_weight"] / (
                (X_transformed["height"] / 100) ** 2
            )
        else:
            X_transformed["bmi"] = np.nan  # Or some imputation strategy

        # Calculate pulse pressure
        if (
            "systolic_bp" in X_transformed.columns
            and "diastolic_bp" in X_transformed.columns
        ):
            X_transformed["pulse_pressure"] = (
                X_transformed["systolic_bp"] - X_transformed["diastolic_bp"]
            )

        # Calculate mean arterial pressure (MAP)
        if (
            "systolic_bp" in X_transformed.columns
            and "diastolic_bp" in X_transformed.columns
        ):
            X_transformed["mean_arterial_pressure"] = (
                X_transformed["systolic_bp"] + 2 * X_transformed["diastolic_bp"]
            ) / 3

        # Calculate rate-pressure product (RPP)
        if (
            "heart_rate" in X_transformed.columns
            and "systolic_bp" in X_transformed.columns
        ):
            X_transformed["rate_pressure_product"] = (
                X_transformed["heart_rate"] * X_transformed["systolic_bp"]
            )

        # Calculate oxygen delivery index (ODI)
        if (
            "heart_rate" in X_transformed.columns
            and "blood_oxygen" in X_transformed.columns
        ):
            X_transformed["oxygen_delivery_index"] = (
                X_transformed["heart_rate"] * X_transformed["blood_oxygen"]
            )

        # Create interaction features
        if "age" in X_transformed.columns and "bmi" in X_transformed.columns:
            X_transformed["age_bmi_interaction"] = (
                X_transformed["age"] * X_transformed["bmi"]
            )

        if (
            "heart_rate" in X_transformed.columns
            and "mean_arterial_pressure" in X_transformed.columns
        ):
            X_transformed["heart_rate_bp_interaction"] = (
                X_transformed["heart_rate"] * X_transformed["mean_arterial_pressure"]
            )

        # Create polynomial features for key vitals
        for vital in ["heart_rate", "blood_oxygen", "body_temp"]:
            if vital in X_transformed.columns:
                X_transformed[f"{vital}_squared"] = X_transformed[vital] ** 2

        # Calculate z-scores for numeric features
        numeric_columns = X_transformed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_transformed[col].std() != 0:  # Avoid division by zero
                X_transformed[f"{col}_zscore"] = (
                    X_transformed[col] - X_transformed[col].mean()
                ) / X_transformed[col].std()

        # Create time-based features
        if "day_of_week" in X_transformed.columns:
            X_transformed["is_weekend"] = (
                X_transformed["day_of_week"].isin([5, 6]).astype(int)
            )

        if "hour" in X_transformed.columns:
            X_transformed["is_night"] = (
                (X_transformed["hour"] >= 22) | (X_transformed["hour"] < 6)
            ).astype(int)

        return X_transformed


# Usage example
if __name__ == "__main__":
    # Assume we have preprocessed data
    preprocessed_data = pd.read_csv("../../data/processed/preprocessed_data.csv")

    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.transform(preprocessed_data)

    print("Feature engineering completed.")
    print(f"Original features: {preprocessed_data.columns.tolist()}")
    print(f"Engineered features: {engineered_data.columns.tolist()}")
    print(
        f"Number of new features added: {len(engineered_data.columns) - len(preprocessed_data.columns)}"
    )
