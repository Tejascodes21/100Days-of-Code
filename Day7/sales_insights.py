# sales_insights.py
# Day 7 - Sales Data Insights CLI using Pandas & NumPy

import pandas as pd
import numpy as np

# Load dataset
try:
    df = pd.read_csv("sales_data.csv")
    print("Sales data loaded successfully!\n")
except FileNotFoundError:
    print("Error: 'sales_data.csv' not found. Please make sure the file exists.")
    exit()

# Add a computed column: Total Sales
df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

# Display basic info
print("Dataset Overview:")
print(df.head(), "\n")

# Basic summary statistics
print("Basic Statistics:\n")
print(df.describe(), "\n")

# Grouping & aggregation
region_sales = df.groupby("Region")["TotalSales"].sum().sort_values(ascending=False)
print("Total Sales by Region:\n", region_sales, "\n")

product_sales = df.groupby("Product")["TotalSales"].sum().sort_values(ascending=False)
print("Top Selling Products:\n", product_sales, "\n")

# Using NumPy for additional analytics
avg_sale = np.mean(df["TotalSales"])
highest_sale = np.max(df["TotalSales"])
lowest_sale = np.min(df["TotalSales"])

print("Advanced Insights:")
print(f"Average Sale Value: ₹{avg_sale:,.2f}")
print(f"Highest Single Order Value: ₹{highest_sale:,.2f}")
print(f"Lowest Single Order Value: ₹{lowest_sale:,.2f}\n")

# Handling missing data check
if df.isnull().values.any():
    print("⚠️ Missing values found — cleaning data...\n")
    df = df.fillna(df.mean(numeric_only=True))
else:
    print("No missing values detected.\n")

# Export cleaned and analyzed data
df.to_csv("sales_data_cleaned.csv", index=False)
print("Cleaned dataset exported as 'sales_data_cleaned.csv'")
