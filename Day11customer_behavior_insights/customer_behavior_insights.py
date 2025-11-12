# Day 11 — Adv EDA: Customer Behavior Insights Dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats
import os

# Step 1: Load Dataset
data = {
    'Customer ID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Age': [25, 30, 35, 40, 29, 33, 38, 50],
    'Gender': ['Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'City': ['Pune', 'Mumbai', 'Delhi', 'Delhi', 'Pune', 'Mumbai', 'Chennai', 'Pune'],
    'Purchase Amount': [25000, 18000, 22000, 27000, 29000, 15000, 30000, 45000],
    'Join Date': ['2022-05-10', '2022-06-14', '2022-07-22', '2022-08-01', '2022-08-20', '2022-09-10', '2022-10-09', '2022-12-01']
}

df = pd.DataFrame(data)
df['Join Date'] = pd.to_datetime(df['Join Date'], errors='coerce')


# Step 2: Data Summary
print("\n=== Dataset Overview ===")
print(df.info())
print("\n=== Summary Statistics ===")
print(df.describe())


# Step 3: Correlation & Feature Analysis

print("\n=== Correlation Matrix ===")
corr = df.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# Step 4: Outlier Detection using Isolation Forest

iso = IsolationForest(contamination=0.1, random_state=42)
df['Outlier'] = iso.fit_predict(df[['Purchase Amount']])
outliers = df[df['Outlier'] == -1]
print("\nDetected Outliers:\n", outliers[['Customer ID', 'Purchase Amount']])


# Step 5: Time-based Insights

df['Month'] = df['Join Date'].dt.month_name()
monthly_sales = df.groupby('Month')['Purchase Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
monthly_sales.plot(kind='bar', color='skyblue')
plt.title("Total Purchase Amount by Month")
plt.xlabel("Month")
plt.ylabel("Total Purchase (₹)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step 6: Statistical Comparison — Gender vs Purchase

male_purchases = df[df['Gender'] == 'Male']['Purchase Amount']
female_purchases = df[df['Gender'] == 'Female']['Purchase Amount']
t_stat, p_val = stats.ttest_ind(male_purchases, female_purchases)

print(f"\n T-Test between Male and Female Purchases: t={t_stat:.2f}, p={p_val:.3f}")
if p_val < 0.05:
    print(" Significant difference between male and female spending habits.")
else:
    print(" No significant difference in spending habits by gender.")


# Step 7: Visual Insights — Distribution & Relationships

sns.set(style="whitegrid")


plt.figure(figsize=(7, 4))
sns.histplot(df['Purchase Amount'], kde=True, color='teal')
plt.title("Distribution of Purchase Amounts")
plt.tight_layout()
plt.show()

# city-wise Purchase Boxplot
plt.figure(figsize=(7, 4))
sns.boxplot(x='City', y='Purchase Amount', data=df, palette='pastel')
plt.title("City-wise Purchase Distribution")
plt.tight_layout()
plt.show()

# Pairplot for numeric features
sns.pairplot(df[['Age', 'Purchase Amount']], diag_kind='kde')
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()


# Step 8: Save Results

os.makedirs("reports", exist_ok=True)
df.to_csv("reports/customer_insights_with_outliers.csv", index=False)
print("\n Analysis complete! Report saved to 'reports/customer_insights_with_outliers.csv'")
