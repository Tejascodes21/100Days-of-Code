# Day 8 — Sales Visualization Dashboard
# Concepts: Data Visualization with Matplotlib and Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#1
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Region': ['North', 'South', 'East', 'West', 'North', 'South'],
    'Category': ['Electronics', 'Clothing', 'Furniture', 'Electronics', 'Clothing', 'Furniture'],
    'Sales': [25000, 18000, 22000, 27000, 30000, 21000],
    'Profit': [5000, 2500, 3000, 4800, 5200, 3100]
}

df = pd.DataFrame(data)


os.makedirs("charts", exist_ok=True)


#1 — Bar chart: Sales by Region

plt.figure(figsize=(8, 5))
sns.barplot(x='Region', y='Sales', data=df, palette='viridis')
plt.title('Sales by Region')
plt.xlabel('Region')
plt.ylabel('Sales (₹)')
plt.tight_layout()
plt.savefig("charts/sales_by_region.png")
plt.show()

# 2 — Line chart: Monthly Sales Trend

plt.figure(figsize=(8, 5))
plt.plot(df['Month'], df['Sales'], marker='o', color='royalblue')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales (₹)')
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/monthly_sales_trend.png")
plt.show()


# 3 — Pie chart: Category-wise Sales Share

category_sales = df.groupby('Category')['Sales'].sum()
plt.figure(figsize=(6, 6))
plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
plt.title('Category-wise Sales Distribution')
plt.tight_layout()
plt.savefig("charts/category_sales_pie.png")
plt.show()

#4 — Heatmap: Correlation Matrix

plt.figure(figsize=(6, 4))
sns.heatmap(df[['Sales', 'Profit']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Sales and Profit')
plt.tight_layout()
plt.savefig("charts/sales_profit_correlation.png")
plt.show()

print("All charts generated and saved in 'charts/' folder successfully!")
