from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted")

    # Check for missing or mismatched data
    if len(y_true) != len(y_pred_proba):
        print(
            f"Warning: Length of y_true ({len(y_true)}) does not match length of y_pred_proba ({len(y_pred_proba)})"
        )
        print("y_true:", y_true)
        print("y_pred_proba shape:", y_pred_proba.shape)
        return metrics

    # Extract unique classes from y_true
    unique_classes = np.unique(y_true)
    num_classes = len(unique_classes)

    # Check if the number of columns in y_pred_proba matches the number of unique classes
    if y_pred_proba.shape[1] != num_classes:
        print(
            f"Warning: Number of classes in y_true ({num_classes}) does not match number of columns in y_pred_proba ({y_pred_proba.shape[1]})"
        )
        print("y_true classes:", unique_classes)
        print("y_pred_proba shape:", y_pred_proba.shape)
        return metrics

    # Print column names of y_pred_proba if it's a DataFrame
    if isinstance(y_pred_proba, pd.DataFrame):
        print("Column names in y_pred_proba:", y_pred_proba.columns.tolist())
    else:
        print("y_pred_proba shape:", y_pred_proba.shape)

    # Calculate ROC AUC for multi-class
    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_pred_proba, multi_class="ovr", average="weighted"
        )
    except ValueError as e:
        print(f"ROC AUC calculation error: {e}")
        metrics["roc_auc"] = None  # Handle error gracefully

    # Confusion matrix with specified labels
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=unique_classes
    )

    return metrics


def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, classes):
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == classes[i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(
            fpr[i], tpr[i], label=f"ROC curve (AUC = {roc_auc[i]:.2f}) for {classes[i]}"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
