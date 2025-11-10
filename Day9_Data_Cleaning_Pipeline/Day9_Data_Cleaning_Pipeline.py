# Day 9 — Customer Data Cleaning Pipeline
# Concepts: Data Cleaning,Preprocessing,Pandas Operations

import pandas as pd
import numpy as np


# 1️ Sample Dataset (replace with a CSV file)

data = {
    'Customer ID': [101, 102, 103, 104, 105, 106, 107, 107],
    'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Frank', 'Grace', 'Grace'],
    'Age': [25, 30, np.nan, 45, 29, 33, 38, 38],
    'Gender': ['Female', 'Male', 'Male', 'Male', None, 'Male', 'Female', 'Female'],
    'City': ['Pune', 'Mumbai', 'Delhi', 'Delhi', 'Pune', 'Mumbai', 'Chennai', 'Chennai'],
    'Purchase Amount': [25000, 18000, 22000, None, 27000, 15000, 30000, 30000],
    'Join Date': ['2022-05-10', '2022-06-14', '2022/07/22', '2022-08-01', 'N/A', '2022-10-09', '2022-10-09', '2022-10-09']
}

df = pd.DataFrame(data)
print(" Raw Dataset:\n", df, "\n")


# 2️ Clean Column Names

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


# 3️ Handle Missing Values

df['name'].fillna('Unknown', inplace=True)
df['gender'].fillna('Not Specified', inplace=True)
df['purchase_amount'].fillna(df['purchase_amount'].mean(), inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)


# 4️ Remove Duplicates

df.drop_duplicates(inplace=True)

# 5️ Fix Data Types

df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')


# 6️ Detect & Remove Outliers (IQR Method for 'purchase_amount')

Q1 = df['purchase_amount'].quantile(0.25)
Q3 = df['purchase_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['purchase_amount'] >= lower_bound) & (df['purchase_amount'] <= upper_bound)]


# 7️ Standardize String Data

df['city'] = df['city'].str.title()
df['gender'] = df['gender'].str.title()


# 8️ Save Cleaned Data

df.to_csv("cleaned_customer_data.csv", index=False)
print(" Cleaned Data Saved as 'cleaned_customer_data.csv'\n")


# 9️ Final Summary

print(" Cleaned Dataset:\n", df)
print("\n Missing Values Summary:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
