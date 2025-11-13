# Day 12 MP 2 — Sales Features + PCA Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# Sample Sales Dataset
df = pd.DataFrame({
    "OrderID": [1, 2, 3, 4, 5, 6],
    "Units": [3, 5, 2, 8, 1, 4],
    "UnitPrice": [700, 1500, 1200, 500, 2000, 650],
    "Category": ["Electronics", "Clothing", "Electronics", "Grocery", "Clothing", "Grocery"],
    "OrderDate": ["2024-01-10", "2024-01-12", "2024-01-14", "2024-02-10", "2024-02-15", "2024-03-20"]
})

df["OrderDate"] = pd.to_datetime(df["OrderDate"])


# Feature Engineering
df["Revenue"] = df["Units"] * df["UnitPrice"]

#Extracting month as a feature
df["Month"] = df["OrderDate"].dt.month

# Price-based category
df["PriceCategory"] = pd.cut(df["UnitPrice"], bins=[0, 800, 1500, 2500],
                             labels=["Low", "Medium", "High"])

# Preprocessing

num_cols = ["Units", "UnitPrice", "Revenue", "Month"]
cat_cols = ["Category", "PriceCategory"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols)
])

# PCA
pca = PCA(n_components=3)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("pca", pca)
])

pca_data = pipeline.fit_transform(df)


# PCA Variance Plot

plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("PCA Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.tight_layout()
plt.savefig("Day12_PCA_Variance.png")
plt.show()

#save output
pd.DataFrame(pca_data).to_csv("Day12_Sales_PCA.csv", index=False)

print("Completed: Data saved & PCA plot generated!")
