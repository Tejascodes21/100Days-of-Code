# Day 12 Proj-1 Customer Spending (Feature Engineering)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dataset (small realistic sample)

df = pd.DataFrame({
    "CustomerID": [101, 102, 103, 104, 105],
    "Age": [24, 32, 45, 28, 36],
    "Gender": ["Female", "Male", "Male", "Female", "Male"],
    "City": ["Pune", "Mumbai", "Delhi", "Pune", "Chennai"],
    "PurchaseAmount": [25000, 18000, 32000, 22000, 27000]
})


# Feature Engineering
# spend per age (intuitive feature)
df["SpendPerAge"] = df["PurchaseAmount"] / df["Age"]

# Age groups
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 25, 35, 50], 
                        labels=["Young", "Adult", "Senior"])

# City purchase frequency
city_counts = df["City"].value_counts()
df["CityFrequency"] = df["City"].map(city_counts)


# preprocessing pipeline

numeric_cols = ["Age", "PurchaseAmount", "SpendPerAge", "CityFrequency"]
categorical_cols = ["Gender", "City", "AgeGroup"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(), categorical_cols)
])

pca = PCA(n_components=2)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("pca", pca)
])

transformed = pipeline.fit_transform(df)

# save results
pd.DataFrame(transformed).to_csv("Day12_Customer_PCA.csv", index=False)

print("Proj 1 Completed: Day12_Customer_PCA.csv created!")
