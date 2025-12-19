import json

with open("registry/model_registry.json", "r") as f:
    registry = json.load(f)

for model in registry:
    model["status"] = "archived"

best_model = max(registry, key=lambda x: x["accuracy"])
best_model["status"] = "production"

with open("registry/model_registry.json", "w") as f:
    json.dump(registry, f, indent=4)

print(f"Model {best_model['version']} promoted to production")
