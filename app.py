from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"

# Load model
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

FEATURES = [
    "Age at enrollment",
    "Gender",
    "Course",
    "Daytime/evening attendance",
    "Scholarship holder",
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "approval_ratio",
    "evaluation_ratio"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        values = []
        for feature in FEATURES:
            values.append(float(request.form[feature]))

        input_df = pd.DataFrame([values], columns=FEATURES)
        pred_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

