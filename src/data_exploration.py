import pandas as pd
import seaborn as sns

import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt


def explore_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Visualizing the distribution of conditions
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Condition", data=data)
    plt.title("Condition Distribution")
    plt.xlabel("Condition")
    plt.ylabel("Count")
    plt.show()

    # Visualizing relationships between features
    plt.figure(figsize=(12, 10))
    sns.pairplot(data, hue="Condition")
    plt.show()


if __name__ == "__main__":
    explore_data("data/dummy_data.csv")
