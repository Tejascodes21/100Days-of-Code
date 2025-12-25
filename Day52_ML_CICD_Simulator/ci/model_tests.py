from sklearn.metrics import accuracy_score

def validate_model(model, X_val, y_val, threshold=0.5):
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    print(f"Validation Accuracy: {accuracy:.2f}")

    if accuracy < threshold:
        raise ValueError("Model failed baseline performance check")

    print("Model passed baseline performance check")
    return accuracy
