import json
from datetime import datetime

registry_path = "registry/model_registry.json"

model_entry = {
    "version": "v2",
    "accuracy": 0.84,
    "dataset_version": "v1",
    "created_at": datetime.now().isoformat(),
    "status": "staging"
}

try:
    with open(registry_path, "r") as f:
        registry = json.load(f)
except FileNotFoundError:
    registry = []

registry.append(model_entry)

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=4)

print("Model registered successfully")