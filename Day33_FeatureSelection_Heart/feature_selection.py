# Day 33 Feature Selection Study - Heart Disease Prediction

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("heart.csv")

X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

results = {}


# Baseline Model

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, model.predict(X_test))
results['Baseline'] = baseline_acc

print(f"Baseline Accuracy: {baseline_acc:.3f}")



# 1️ Variance Threshold

vt = VarianceThreshold(threshold=0.0)
X_vt = vt.fit_transform(X)
selected_features_vt = X.columns[vt.get_support()]

print("\nVariance Threshold selected features:")
print(selected_features_vt.tolist())

X_train_vt, X_test_vt, y_train, y_test = train_test_split(
    X_vt, y, test_size=0.3, random_state=42
)

model.fit(X_train_vt, y_train)
results['Variance Threshold'] = accuracy_score(y_test, model.predict(X_test_vt))



# 2 SelectKBest (ANOVA F-test)

skb = SelectKBest(score_func=f_classif, k=5)
X_skb = skb.fit_transform(X, y)
selected_features_skb = X.columns[skb.get_support()]

print("\nSelectKBest selected features:")
print(selected_features_skb.tolist())

X_train_skb, X_test_skb, y_train, y_test = train_test_split(
    X_skb, y, test_size=0.3, random_state=42
)

model.fit(X_train_skb, y_train)
results['SelectKBest'] = accuracy_score(y_test, model.predict(X_test_skb))



# 3 Recursive Feature Elimination (RFE)

rfe = RFE(estimator=LogisticRegression(max_iter=300), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
selected_features_rfe = X.columns[rfe.get_support()]

print("\nRFE selected features:")
print(selected_features_rfe.tolist())

X_train_rfe, X_test_rfe, y_train, y_test = train_test_split(
    X_rfe, y, test_size=0.3, random_state=42
)

model.fit(X_train_rfe, y_train)
results['RFE'] = accuracy_score(y_test, model.predict(X_test_rfe))



print("\nModel Accuracy Comparison:")
print(results)

plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Feature Selection Performance Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
