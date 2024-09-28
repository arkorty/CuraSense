import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def preprocess_sensor_data(self, sensor_data):
        # Convert timestamp to datetime
        sensor_data["timestamp"] = pd.to_datetime(sensor_data["timestamp"])

        # Extract time-based features
        sensor_data["hour"] = sensor_data["timestamp"].dt.hour
        sensor_data["day_of_week"] = sensor_data["timestamp"].dt.dayofweek

        # Calculate rolling statistics
        numeric_columns = [
            "heart_rate",
            "blood_oxygen",
            "systolic_bp",
            "diastolic_bp",
            "body_temp",
            "body_weight",
        ]
        for col in numeric_columns:
            sensor_data[f"{col}_rolling_mean"] = (
                sensor_data.groupby("patient_id")[col]
                .rolling(window=5)
                .mean()
                .reset_index(0, drop=True)
            )
            sensor_data[f"{col}_rolling_std"] = (
                sensor_data.groupby("patient_id")[col]
                .rolling(window=5)
                .std()
                .reset_index(0, drop=True)
            )

        # Impute missing values
        sensor_data_imputed = pd.DataFrame(
            self.imputer.fit_transform(sensor_data[numeric_columns]),
            columns=numeric_columns,
            index=sensor_data.index,
        )

        # Scale numeric features
        sensor_data_scaled = pd.DataFrame(
            self.scaler.fit_transform(sensor_data_imputed),
            columns=numeric_columns,
            index=sensor_data.index,
        )

        # Combine processed features
        sensor_data_processed = pd.concat(
            [sensor_data_scaled, sensor_data[["patient_id", "hour", "day_of_week"]]],
            axis=1,
        )

        return sensor_data_processed

    def preprocess_admission_data(self, admission_data):
        # Convert admission_date to datetime
        admission_data["admission_date"] = pd.to_datetime(
            admission_data["admission_date"]
        )

        # Calculate length of stay
        admission_data["length_of_stay"] = (
            pd.Timestamp.now() - admission_data["admission_date"]
        ).dt.days

        # Encode categorical variables
        categorical_columns = ["condition", "severity", "gender"]
        encoded_features = self.encoder.fit_transform(
            admission_data[categorical_columns]
        )
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=self.encoder.get_feature_names_out(categorical_columns),
            index=admission_data.index,
        )

        # Combine processed features
        admission_data_processed = pd.concat(
            [admission_data[["patient_id", "age", "length_of_stay"]], encoded_df],
            axis=1,
        )

        return admission_data_processed

    def merge_data(self, sensor_data_processed, admission_data_processed):
        merged_data = pd.merge(
            sensor_data_processed, admission_data_processed, on="patient_id"
        )
        return merged_data

    def split_data(self, merged_data, target_column, test_size=0.2, random_state=42):
        X = merged_data.drop(columns=[target_column])
        y = merged_data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Usage example
if __name__ == "__main__":
    sensor_data = pd.read_csv("../../data/raw/sensor_data.csv")
    admission_data = pd.read_csv("../../data/raw/admission_data.csv")

    preprocessor = DataPreprocessor()
    sensor_data_processed = preprocessor.preprocess_sensor_data(sensor_data)
    admission_data_processed = preprocessor.preprocess_admission_data(admission_data)

    merged_data = preprocessor.merge_data(
        sensor_data_processed, admission_data_processed
    )

    # Assuming we want to predict 'severity' as our target variable
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        merged_data, target_column="severity"
    )

    print("Data preprocessing completed.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
