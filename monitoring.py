
import csv
import os
from datetime import datetime, timezone

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_logs.csv")

COLUMNS = [
    "timestamp",
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "predicted_species",
    "setosa_probability",
    "versicolor_probability",
    "virginica_probability",
]


def log_prediction(features, predicted_species, probabilities):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)

        if not file_exists or os.path.getsize(LOG_FILE) == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3],
            "predicted_species": predicted_species,
            "setosa_probability": probabilities["setosa"],
            "versicolor_probability": probabilities["versicolor"],
            "virginica_probability": probabilities["virginica"],
        })
