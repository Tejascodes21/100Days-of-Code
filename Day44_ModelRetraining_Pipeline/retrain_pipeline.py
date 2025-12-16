import json
import os
from train_model import train_and_evaluate

OLD_MODEL_PATH = "models/model_v1.pkl"
OLD_METRICS_PATH = "models/metrics_v1.json"

NEW_MODEL_PATH = "models/model_v2.pkl"
NEW_METRICS_PATH = "models/metrics_v2.json"

print(" Starting retraining pipeline...")

new_accuracy = train_and_evaluate(
    data_path="data/new_data.csv",
    model_path=NEW_MODEL_PATH,
    metrics_path=NEW_METRICS_PATH
)

with open(OLD_METRICS_PATH, "r") as f:
    old_accuracy = json.load(f)["accuracy"]

print(f" Old Model Accuracy: {old_accuracy}")
print(f" New Model Accuracy: {new_accuracy}")

if new_accuracy > old_accuracy:
    print(" New model is better — promoting to production")

    os.replace(NEW_MODEL_PATH, OLD_MODEL_PATH)
    os.replace(NEW_METRICS_PATH, OLD_METRICS_PATH)

else:
    print(" New model underperformed — keeping current production model")
