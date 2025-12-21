import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "age": np.random.normal(55, 5, 20).astype(int),
    "cholesterol": np.random.normal(260, 15, 20).astype(int),
    "blood_pressure": np.random.normal(155, 10, 20).astype(int),
    "bmi": np.random.normal(31, 2, 20),
    "smoker": np.random.choice([0, 1], 20)
}

df = pd.DataFrame(data)
df.to_csv("data/inference_logs.csv", index=False)

print("Inference data logged")
