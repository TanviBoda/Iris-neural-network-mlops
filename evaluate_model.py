import joblib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "iris.csv"
MODEL_FILE = BASE_DIR / "iris_model.keras"
SCALER_FILE = BASE_DIR / "scaler.pkl"
REPORT_FILE = BASE_DIR / "evaluation_report.json"

# Load dataset
df = pd.read_csv(DATASET_FILE)

feature_columns = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

X = df[feature_columns].to_numpy(dtype=float)
label_column = next(
    col for col in df.columns
    if col.lower() in ["species", "variety", "class", "target"]
)
encoder = LabelEncoder()
y = encoder.fit_transform(df[label_column].astype(str))

# Re-create the same held-out split used in training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)

# Fit scaler only on training data to avoid data leakage
scaler = joblib.load(SCALER_FILE)
X_test_scaled = scaler.transform(X_test)

# Load the saved model and evaluate it
model = tf.keras.models.load_model(MODEL_FILE)
probabilities = model.predict(X_test_scaled, verbose=0)
predictions = np.argmax(probabilities, axis=1)

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0,
)

results = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=np.arange(len(encoder.classes_))
    ).tolist(),
    "class_names": encoder.classes_.tolist(),
    "test_samples": int(len(y_test)),
}

REPORT_FILE.write_text(json.dumps(results, indent=4), encoding="utf-8")

print("===== MODEL EVALUATION =====")
print(f"Test samples: {len(y_test)}")
print(f"Accuracy: {results['accuracy']:.2%}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0,
))
print("Confusion matrix:")
print(np.array(results["confusion_matrix"]))
print(f"\nReport saved to: {REPORT_FILE}")
