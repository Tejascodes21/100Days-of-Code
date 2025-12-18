import pandas as pd

runs = pd.read_csv("experiments/runs.csv")

print(" All Experiment Runs:")
print(runs[["run_id", "accuracy", "f1_score"]])

best_run = runs.sort_values("f1_score", ascending=False).iloc[0]

print("\nBest Experiment:")
print(best_run)
