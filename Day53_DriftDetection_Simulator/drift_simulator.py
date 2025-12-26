
# Day 53 - Drift Detection Simulator
import pandas as pd

train_df = pd.read_csv("data/train_data.csv")
prod_df = pd.read_csv("data/production_data.csv")

print("\n Training Data Summary:")
print(train_df.describe())

print("\n Production Data Summary:")
print(prod_df.describe())


# Drift Detection Logic

THRESHOLD = 0.15  # 15% mean shift allowed
drift_detected = False

print("\n Drift Analysis Results:\n")

for column in train_df.columns:
    train_mean = train_df[column].mean()
    prod_mean = prod_df[column].mean()

    shift = abs(prod_mean - train_mean) / train_mean

    print(f"Feature: {column}")
    print(f" Training Mean   : {train_mean:.2f}")
    print(f" Production Mean : {prod_mean:.2f}")
    print(f" Mean Shift      : {shift:.2%}")

    if shift > THRESHOLD:
        print(" Drift Detected!")
        drift_detected = True
    else:
        print(" No Significant Drift")

    print("-" * 40)


# Final Decision

if drift_detected:
    print("\n ACTION REQUIRED:")
    print("Data drift detected.Consider retraining or deeper analysis.")
else:
    print("\n SYSTEM STABLE:")
    print("No major drift detected.Continue monitoring.")
