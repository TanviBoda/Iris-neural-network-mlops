
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate the existing prediction log
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "prediction_logs.csv"
GRAPH_FILE = BASE_DIR / "prediction_monitoring.png"

# Load logged predictions
if not LOG_FILE.exists():
    raise FileNotFoundError(
        "prediction_logs.csv was not found. Make a prediction through the API first."
    )

df = pd.read_csv(LOG_FILE)

if df.empty:
    raise ValueError("The prediction log is empty.")

# Count predictions by species
species_counts = df["predicted_species"].value_counts()

# Calculate confidence of each recorded prediction
probability_columns = [
    "setosa_probability",
    "versicolor_probability",
    "virginica_probability",
]

df["confidence"] = df[probability_columns].max(axis=1)

print("===== IRIS MODEL MONITORING REPORT =====")
print("Total predictions:", len(df))
print("\nPredictions by species:")
print(species_counts.to_string())
print(f"\nAverage prediction confidence: {df['confidence'].mean():.2%}")

# Generate a bar chart
species_counts.reindex(
    ["setosa", "versicolor", "virginica"], fill_value=0
).plot(kind="bar", rot=0)

plt.title("Iris Model - Prediction Counts")
plt.xlabel("Predicted Species")
plt.ylabel("Number of Predictions")
plt.tight_layout()
plt.savefig(GRAPH_FILE, dpi=150)
plt.close()

print(f"\nMonitoring graph saved to: {GRAPH_FILE}")
