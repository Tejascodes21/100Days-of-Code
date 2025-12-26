import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


ref_df = pd.read_csv("data/reference_data.csv")
cur_df = pd.read_csv("data/current_data.csv")


# PSI calculation

def calculate_psi(ref, cur, buckets=10):
    breakpoints = np.percentile(ref, np.linspace(0, 100, buckets + 1))
    
    ref_counts = np.histogram(ref, breakpoints)[0] / len(ref)
    cur_counts = np.histogram(cur, breakpoints)[0] / len(cur)

    psi = np.sum(
        (ref_counts - cur_counts) * np.log((ref_counts + 1e-6) / (cur_counts + 1e-6))
    )
    return psi


# Drift Monitoring

print("\n Drift Monitoring Report\n")

for column in ref_df.columns:
    ref_col = ref_df[column]
    cur_col = cur_df[column]

    psi_value = calculate_psi(ref_col, cur_col)
    ks_stat, p_value = ks_2samp(ref_col, cur_col)

    print(f"Feature: {column}")
    print(f"  PSI Score : {psi_value:.4f}")
    print(f"  KS p-value: {p_value:.4f}")

    if psi_value > 0.25:
        print("  High data drift detected")
    elif psi_value > 0.1:
        print("  Moderate data drift detected")
    else:
        print("  No significant drift")

    print("-" * 40)
