from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_reduced.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# Load model and encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

FEATURES = [
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 1st sem (evaluations)",
    "Age at enrollment",
    "Tuition fees up to date"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # Read and validate values
        sem2_approved = max(0, int(request.form["sem2_approved"]))
        sem2_grade_10 = max(0, float(request.form["sem2_grade"]))
        sem1_approved = max(0, int(request.form["sem1_approved"]))
        sem1_grade_10 = max(0, float(request.form["sem1_grade"]))
        sem2_eval = max(0, int(request.form["sem2_eval"]))
        sem1_eval = max(0, int(request.form["sem1_eval"]))
        age = max(0, int(request.form["age"]))
        fees = int(request.form["fees"])

        # Convert grade scale 1–10 → 1–20
        sem2_grade_20 = sem2_grade_10 * 2
        sem1_grade_20 = sem1_grade_10 * 2

        input_data = [[
            sem2_approved,
            sem2_grade_20,
            sem1_approved,
            sem1_grade_20,
            sem2_eval,
            sem1_eval,
            age,
            fees
        ]]

        input_df = pd.DataFrame(input_data, columns=FEATURES)

        pred_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
