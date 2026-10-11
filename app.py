
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from tensorflow import keras
import numpy as np
import joblib

app = FastAPI(
    title="Iris AI",
    description="Iris flower classification using a neural network",
    version="1.0.0"
)

model = keras.models.load_model("iris_model.keras")
scaler = joblib.load("scaler.pkl")

species_names = ["setosa", "versicolor", "virginica"]


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


class IrisPrediction(BaseModel):
    predicted_species: str
    probabilities: dict[str, float]


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris AI | Flower Classifier</title>
<style>
:root {
  --ink: #202b29;
  --muted: #75817d;
  --green: #286653;
  --mint: #e8f2ec;
  --line: #e5eae6;
  --paper: #ffffff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  background: #f5f7f4;
  color: var(--ink);
}
header {
  height: 76px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 7%;
  background: white;
  border-bottom: 1px solid var(--line);
}
.brand { display: flex; align-items: center; gap: 12px; font-weight: 750; }
.logo {
  width: 40px; height: 40px; border-radius: 13px;
  background: var(--green); color: white;
  display: grid; place-items: center; font-size: 22px;
}
.tag {
  font-size: 12px; color: var(--green); background: var(--mint);
  padding: 8px 12px; border-radius: 30px; font-weight: 650;
}
main { max-width: 1080px; margin: 46px auto; padding: 0 22px; }
.hero { margin-bottom: 28px; }
.eyebrow {
  color: var(--green); font-size: 12px; letter-spacing: 2px;
  font-weight: 800; text-transform: uppercase;
}
h1 { font-size: clamp(30px, 4vw, 44px); letter-spacing: -1.5px; margin: 12px 0; }
.hero p { color: var(--muted); line-height: 1.7; max-width: 620px; margin: 0; }
.layout { display: grid; grid-template-columns: 1fr 0.9fr; gap: 22px; }
.card {
  background: var(--paper); border: 1px solid var(--line);
  border-radius: 20px; padding: 28px;
  box-shadow: 0 8px 28px rgba(32,43,41,.035);
}
.card h2 { margin: 0 0 8px; font-size: 20px; }
.sub { color: var(--muted); font-size: 13px; line-height: 1.6; margin: 0 0 24px; }
.fields { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
label { display: block; font-size: 13px; font-weight: 650; margin-bottom: 9px; }
input {
  width: 100%; border: 1px solid #dce3dd; border-radius: 10px;
  padding: 13px 12px; font-size: 15px; color: var(--ink);
  outline: none; background: #fff;
}
input:focus { border-color: var(--green); box-shadow: 0 0 0 3px #28665318; }
.unit { color: var(--muted); font-size: 11px; font-weight: 400; }
.actions { display: flex; gap: 12px; margin-top: 24px; }
button {
  cursor: pointer; border: 0; border-radius: 11px;
  padding: 14px 18px; font-size: 14px; font-weight: 700;
}
.primary { background: var(--green); color: white; flex: 1; }
.primary:hover { background: #204f41; }
.secondary { background: #eef2ee; color: var(--ink); }
.result { min-height: 340px; display: flex; flex-direction: column; }
.result-top { display: flex; justify-content: space-between; align-items: center; }
.status { font-size: 11px; color: var(--green); background: var(--mint); padding: 7px 10px; border-radius: 20px; }
.empty {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center; padding: 24px 8px;
}
.flower { font-size: 60px; margin-bottom: 14px; }
.empty h3 { margin: 0 0 9px; font-size: 18px; }
.empty p { max-width: 260px; color: var(--muted); font-size: 13px; line-height: 1.7; margin: 0; }
#output { display: none; margin-top: 25px; }
.species { font-size: 29px; font-weight: 800; text-transform: capitalize; margin: 8px 0; }
.confidence { color: var(--muted); font-size: 13px; margin-bottom: 25px; }
.bar-row { margin: 17px 0; }
.bar-label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px; text-transform: capitalize; }
.track { height: 8px; background: #edf0ed; border-radius: 20px; overflow: hidden; }
.fill { height: 100%; background: #5a947b; border-radius: 20px; transition: width .4s; }
.note { color: var(--muted); font-size: 12px; line-height: 1.6; margin-top: 25px; }
#error { display: none; color: #a13737; background: #fff0f0; padding: 12px; border-radius: 10px; font-size: 13px; margin-top: 15px; }
footer { text-align: center; padding: 24px; color: var(--muted); font-size: 12px; }
@media(max-width: 760px) {
  header { padding: 0 5%; }
  main { margin: 30px auto; }
  .layout { grid-template-columns: 1fr; }
  .card { padding: 22px; }
  .result { min-height: 280px; }
}
@media(max-width: 380px) { .fields { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div class="brand"><div class="logo">✿</div><span>Iris AI</span></div>
  <span class="tag">● Neural Network</span>
</header>

<main>
  <section class="hero">
    <div class="eyebrow">Machine Learning · Classification</div>
    <h1>Discover your Iris flower.</h1>
    <p>Enter four flower measurements and let a trained neural network predict the species. Explore the model's confidence across all three classes.</p>
  </section>

  <section class="layout">
    <div class="card">
      <h2>Flower measurements</h2>
      <p class="sub">Enter the dimensions in centimeters (cm).</p>
      <form id="irisForm">
        <div class="fields">
          <div>
            <label for="sl">Sepal length <span class="unit">/ cm</span></label>
            <input id="sl" type="number" min="0.1" max="10" step="0.1" value="5.1" required>
          </div>
          <div>
            <label for="sw">Sepal width <span class="unit">/ cm</span></label>
            <input id="sw" type="number" min="0.1" max="10" step="0.1" value="3.5" required>
          </div>
          <div>
            <label for="pl">Petal length <span class="unit">/ cm</span></label>
            <input id="pl" type="number" min="0.1" max="10" step="0.1" value="1.4" required>
          </div>
          <div>
            <label for="pw">Petal width <span class="unit">/ cm</span></label>
            <input id="pw" type="number" min="0.1" max="10" step="0.1" value="0.2" required>
          </div>
        </div>
        <div class="actions">
          <button class="primary" id="predictBtn" type="submit">Predict species →</button>
          <button class="secondary" type="reset">Reset</button>
        </div>
        <div id="error" role="alert"></div>
      </form>
    </div>

    <div class="card result">
      <div class="result-top">
        <h2>Prediction result</h2>
        <span class="status">● Model ready</span>
      </div>
      <p class="sub">Your classification will appear here.</p>
      <div class="empty" id="empty">
        <div class="flower">🌱</div>
        <h3>Ready to identify</h3>
        <p>Submit the measurements to see the predicted species and its probability scores.</p>
      </div>
      <div id="output" aria-live="polite">
        <div class="eyebrow">Predicted species</div>
        <div class="species" id="species">Setosa</div>
        <div class="confidence" id="confidence"></div>
        <div id="bars"></div>
        <p class="note">Probabilities are the model's estimates, not a guarantee of correctness.</p>
      </div>
    </div>
  </section>
</main>
<footer>Iris AI · TensorFlow neural network · FastAPI</footer>

<script>
const form = document.getElementById("irisForm");
const btn = document.getElementById("predictBtn");
const errorBox = document.getElementById("error");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.style.display = "none";
  btn.disabled = true;
  btn.textContent = "Analyzing...";

  const payload = {
    sepal_length: Number(document.getElementById("sl").value),
    sepal_width: Number(document.getElementById("sw").value),
    petal_length: Number(document.getElementById("pl").value),
    petal_width: Number(document.getElementById("pw").value)
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Prediction failed.");

    document.getElementById("empty").style.display = "none";
    document.getElementById("output").style.display = "block";
    document.getElementById("species").textContent = data.predicted_species;
    const confidence = data.probabilities[data.predicted_species] * 100;
    document.getElementById("confidence").textContent =
      confidence.toFixed(2) + "% model confidence for the predicted species";

    const bars = document.getElementById("bars");
    bars.replaceChildren();
    for (const [name, probability] of Object.entries(data.probabilities)) {
      const percent = probability * 100;
      const row = document.createElement("div");
      row.className = "bar-row";

      const label = document.createElement("div");
      label.className = "bar-label";
      const nameSpan = document.createElement("span");
      nameSpan.textContent = name;
      const valueSpan = document.createElement("span");
      valueSpan.textContent = percent.toFixed(2) + "%";
      label.append(nameSpan, valueSpan);

      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = percent + "%";
      track.append(fill);
      row.append(label, track);
      bars.append(row);
    }
  } catch (error) {
    errorBox.textContent = error.message || "Unable to connect to the API.";
    errorBox.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict species →";
  }
});

form.addEventListener("reset", () => {
  setTimeout(() => {
    document.getElementById("output").style.display = "none";
    document.getElementById("empty").style.display = "flex";
    errorBox.style.display = "none";
  }, 0);
});
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=IrisPrediction)
def predict(data: IrisInput):
    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]], dtype=float)

        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))

        return {
            "predicted_species": species_names[predicted_index],
            "probabilities": {
                species_names[i]: float(predictions[i])
                for i in range(len(predictions))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
