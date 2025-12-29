import pandas as pd
import json

with open("risk_config.json") as f:
    config = json.load(f)


df = pd.read_csv("data_sample.csv")

print("\n ML Privacy & Security Risk Assessment\n")


# 1. PII Detection

pii_found = [col for col in df.columns if col in config["pii_columns"]]

if pii_found:
    print(f" PII detected: {pii_found}")
else:
    print(" No direct PII detected")


# 2. Sensitive Attribute Check

sensitive_found = [
    col for col in df.columns if col in config["sensitive_attributes"]
]

print(f" Sensitive attributes present: {sensitive_found}")


# 3. Feature Allowlist Check

invalid_features = [
    col for col in df.columns
    if col not in config["allowed_features"]
    and col != "loan_default"
]

if invalid_features:
    print(f" Disallowed features detected: {invalid_features}")
else:
    print(" All features comply with allowlist")


# 4. Risk Evaluation

risk_score = 0

if pii_found:
    risk_score += 2
if sensitive_found:
    risk_score += 1
if config["high_risk_use_case"]:
    risk_score += 2

print(f"\n Final Risk Score: {risk_score}")


# 5. Deployment Decision

if risk_score >= 4:
    print(" Deployment blocked — security & privacy risks too high")
elif risk_score >= 2:
    print(" Deployment allowed with strict monitoring & approvals")
else:
    print(" Deployment approved")
