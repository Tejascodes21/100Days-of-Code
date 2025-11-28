import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


df = pd.read_csv("data/heart_disrp.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# SVM -Linear kernel model

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0))
])

linear_model.fit(X_train, y_train)
linear_preds = linear_model.predict(X_test)

print("\n Linear SVM Results")
print("Accuracy:", accuracy_score(y_test, linear_preds))
print(classification_report(y_test, linear_preds))

# SVM -RBF kernel model

rbf_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=2.0, gamma=0.1))
])

rbf_model.fit(X_train, y_train)
rbf_preds = rbf_model.predict(X_test)

print("\n RBF SVM Results")
print("Accuracy:", accuracy_score(y_test, rbf_preds))
print(classification_report(y_test, rbf_preds))


#Confusion matrix comparison
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, linear_preds), annot=True, cmap="Blues", fmt="d")
plt.title("Linear SVM Confusion Matrix")

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, rbf_preds), annot=True, cmap="Greens", fmt="d")
plt.title("RBF SVM Confusion Matrix")

plt.tight_layout()
plt.show()
