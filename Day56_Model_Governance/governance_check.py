import json
import pandas as pd

# Load files
with open("model_card.json") as f:
    model_card = json.load(f)

with open("fairness_report.json") as f:
    fairness_report = json.load(f)

df = pd.read_csv("sample_predictions.csv")

print("\n Model Governance Review\n")

# Model Card Check
required_fields = ["model_name", "version", "intended_use", "limitations"]
missing = [f for f in required_fields if f not in model_card]

if missing:
    print(" Model Card incomplete:", missing)
else:
    print(" Model Card complete")

# Fairness Check
rates = fairness_report["group_metrics"]
difference = abs(rates["M"]["positive_rate"] - rates["F"]["positive_rate"])

print(f" Fairness gap: {difference:.2f}")

if difference > 0.3:
    print(" Bias threshold exceeded — requires review")
else:
    print(" Fairness within acceptable range")

# Final Decision
if difference > 0.3:
    print("\n Model NOT approved for production")
else:
    print("\n Model approved for deployment with monitoring")
