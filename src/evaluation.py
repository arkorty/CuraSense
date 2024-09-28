from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
