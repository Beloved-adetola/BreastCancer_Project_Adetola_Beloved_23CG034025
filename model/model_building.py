import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = data.target   # 0 = malignant, 1 = benign

# 2. Feature selection (exactly 5)
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean concavity"
]

X = df[selected_features]
y = df["diagnosis"]

# 3. Handle missing values (dataset is clean, but required step)
X = X.fillna(X.mean())

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. Evaluation
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# 8. Save model and scaler
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("Model saved successfully!")

# 9. Reload model (proof of persistence)
with open("breast_cancer_model.pkl", "rb") as f:
    loaded_model, loaded_scaler = pickle.load(f)

sample = X_test.iloc[[0]]
sample_scaled = loaded_scaler.transform(sample)
print("Reloaded model prediction:", loaded_model.predict(sample_scaled))
