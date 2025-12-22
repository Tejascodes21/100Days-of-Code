import json
from pipeline.ingest import load_data
from pipeline.preprocess import preprocess
from pipeline.train import train_model
from pipeline.predict import predict

DATA_PATH = "data/loan_data.csv"
METRICS_PATH = "monitoring/metrics.json"

def run():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = preprocess(df)

    accuracy = train_model(X_train, X_test, y_train, y_test)

   
    metrics = {
        "model_version": "v1",
        "accuracy": accuracy
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    # sample prediction
    sample = [30, 55000, 220000, 670]
    result = predict(sample)

    print(" Sample Prediction:", "Default" if result == 1 else "No Default")

if __name__ == "__main__":
    run()
