
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "iris.csv"
LOG_FILE = BASE_DIR / "prediction_logs.csv"

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
REPORT_FILE = BASE_DIR / "drift_report.csv"
results = []

training_data = pd.read_csv(DATASET_FILE)
prediction_data = pd.read_csv(LOG_FILE)

# Map the actual Iris CSV column names to the API log names
column_mapping = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}

training_data = training_data.rename(columns=column_mapping)

missing = [feature for feature in features if feature not in training_data.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}. Available: {training_data.columns.tolist()}"
    )

print("===== IRIS DATA DRIFT REPORT =====")

for feature in features:
    baseline = pd.to_numeric(training_data[feature], errors="coerce").dropna()
    current = pd.to_numeric(prediction_data[feature], errors="coerce").dropna()

    if baseline.empty or current.empty:
        print(f"\n{feature}: insufficient data")
        continue

    baseline_mean = baseline.mean()
    current_mean = current.mean()
    baseline_std = baseline.std()

    if baseline_std == 0:
        shift = 0.0 if current_mean == baseline_mean else float("inf")
    else:
        shift = abs(current_mean - baseline_mean) / baseline_std

    status = "CHECK FOR DRIFT" if shift > 0.5 else "No large mean shift detected"
    results.append({
    "feature": feature,
    "training_mean": baseline_mean,
    "prediction_mean": current_mean,
    "mean_shift_std": shift,
    "status": status
})

    print(f"\nFeature: {feature}")
    print(f"Training mean: {baseline_mean:.3f}")
    print(f"Prediction mean: {current_mean:.3f}")
    print(f"Mean shift: {shift:.2f} standard deviations")
    print(f"Status: {status}")

print("\nNote: This is a simple mean-shift check, not a formal statistical test.")
pd.DataFrame(results).to_csv(REPORT_FILE, index=False)
print(f"\nDrift report saved to: {REPORT_FILE}")