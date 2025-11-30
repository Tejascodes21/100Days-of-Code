import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data/flight_delay.csv")

X = df.drop("delay", axis=1)
y = df["delay"]

numeric_cols = ["temperature", "humidity", "wind_speed", "visibility", "precipitation"]
categorical_cols = ["airline", "origin"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


# Models for Comparison

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "SVM (RBF Kernel)": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []

# 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    
    results.append([name, acc_scores.mean(), auc_scores.mean()])
    print(f"\n{name}:")
    print(f"Accuracy: {acc_scores.mean():.3f}")
    print(f"ROC-AUC: {auc_scores.mean():.3f}")



results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC_AUC"])
results_df = results_df.sort_values(by="ROC_AUC", ascending=False)
print("\n\n Model Leaderboard:\n")
print(results_df)


# Feature Importance (Random Forest)

best_model_name = results_df.iloc[0]["Model"]

if best_model_name == "Random Forest":
    print(f"\n Best Model → {best_model_name} → Showing Feature Importance")

    rf_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    rf_pipeline.fit(X, y)
    
    # Extracting feature names after encoding
    feature_names = (rf_pipeline.named_steps['prep']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_cols))
    
    all_features = np.concatenate([numeric_cols, feature_names])
    importances = rf_pipeline.named_steps['model'].feature_importances_
    
    # Plot Feature importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=all_features)
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()
else:
    print(f"\nBest Model is {best_model_name} — feature imp not supported directly.")
