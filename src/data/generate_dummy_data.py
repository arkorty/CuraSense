import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sensor_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "patient_id": np.repeat(range(1, 101), 10),
        "timestamp": [
            datetime.now() + timedelta(minutes=i) for i in range(num_samples)
        ],
        "heart_rate": np.random.normal(80, 10, num_samples),
        "blood_oxygen": np.random.normal(98, 2, num_samples),
        "systolic_bp": np.random.normal(120, 10, num_samples),
        "diastolic_bp": np.random.normal(80, 8, num_samples),
        "body_temp": np.random.normal(37, 0.5, num_samples),
        "body_weight": np.random.normal(70, 10, num_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv("data/raw/sensor_data.csv", index=False)
    return df


def generate_admission_data(num_patients=100):
    np.random.seed(42)
    conditions = ["Pneumonia", "Sepsis", "Heart Failure", "Stroke", "Trauma"]
    data = {
        "patient_id": range(1, num_patients + 1),
        "admission_date": [
            datetime.now() - timedelta(days=np.random.randint(1, 30))
            for _ in range(num_patients)
        ],
        "condition": np.random.choice(conditions, num_patients),
        "severity": np.random.choice(["Mild", "Moderate", "Severe"], num_patients),
        "age": np.random.randint(18, 90, num_patients),
        "gender": np.random.choice(["Male", "Female"], num_patients),
    }
    df = pd.DataFrame(data)
    df.to_csv("data/raw/admission_data.csv", index=False)
    return df


sensor_data = generate_sensor_data()
admission_data = generate_admission_data()

print("Dummy data generated and saved in the 'data/raw/' directory.")
