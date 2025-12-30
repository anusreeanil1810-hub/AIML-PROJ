from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

FEATURES = [
<<<<<<< HEAD
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
=======
>>>>>>> 86b138d5fd6e1b97c85df3978e4cbe4e094ae479
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

        # SGPA → grade conversion
        sgpa_sem1 = float(request.form["sgpa_sem1"])
        sgpa_sem2 = float(request.form["sgpa_sem2"])

        grade_sem1 = (sgpa_sem1 / 10) * 20
        grade_sem2 = (sgpa_sem2 / 10) * 20

        input_data = {
            "Curricular units 1st sem (enrolled)": int(request.form["Curricular units 1st sem (enrolled)"]),
            "Curricular units 1st sem (evaluations)": int(request.form["Curricular units 1st sem (evaluations)"]),
            "Curricular units 1st sem (approved)": int(request.form["Curricular units 1st sem (approved)"]),
            "Curricular units 1st sem (grade)": grade_sem1,

            "Curricular units 2nd sem (enrolled)": int(request.form["Curricular units 2nd sem (enrolled)"]),
            "Curricular units 2nd sem (evaluations)": int(request.form["Curricular units 2nd sem (evaluations)"]),
            "Curricular units 2nd sem (approved)": int(request.form["Curricular units 2nd sem (approved)"]),
            "Curricular units 2nd sem (grade)": grade_sem2,

            "Age at enrollment": int(request.form["Age at enrollment"]),
            "Tuition fees up to date": int(request.form["Tuition fees up to date"])
        }

        df = pd.DataFrame([input_data], columns=FEATURES)

        prediction_encoded = model.predict(df)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
