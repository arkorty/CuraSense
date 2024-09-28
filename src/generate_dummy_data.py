import pandas as pd
import numpy as np


def generate_dummy_data(num_samples=1000):
    np.random.seed(42)  # For reproducibility

    # Generating random sensor data
    heart_rate = np.random.randint(60, 120, num_samples)  # bpm
    blood_oxygen = np.random.randint(85, 100, num_samples)  # percentage
    blood_pressure_systolic = np.random.randint(90, 180, num_samples)  # mmHg
    blood_pressure_diastolic = np.random.randint(60, 120, num_samples)  # mmHg
    saline_water_level = np.random.uniform(0.0, 100.0, num_samples)  # liters
    body_temp = np.random.uniform(36.0, 40.0, num_samples)  # Celsius
    body_weight = np.random.uniform(50.0, 100.0, num_samples)  # kg

    # Generating random conditions
    conditions = np.random.choice(
        ["Stable", "Critical", "Life-threatening"], num_samples
    )

    # Tentative recovery time for stable condition
    recovery_time = np.where(
        conditions == "Stable", np.random.randint(1, 15, num_samples), None
    )

    # Next steps for critical condition
    next_steps = np.where(
        conditions == "Critical", ["Monitor vital signs"] * num_samples, None
    )

    # Create DataFrame
    data = pd.DataFrame(
        {
            "Heart Rate": heart_rate,
            "Blood Oxygen": blood_oxygen,
            "Blood Pressure Systolic": blood_pressure_systolic,
            "Blood Pressure Diastolic": blood_pressure_diastolic,
            "Saline Water Level": saline_water_level,
            "Body Temperature": body_temp,
            "Body Weight": body_weight,
            "Condition": conditions,
            "Tentative Recovery Time": recovery_time,
            "Next Steps": next_steps,
        }
    )

    # Save the DataFrame to a CSV file
    data.to_csv("data/dummy_data.csv", index=False)


if __name__ == "__main__":
    generate_dummy_data()
