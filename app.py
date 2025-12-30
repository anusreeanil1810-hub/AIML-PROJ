from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your model and label encoder
MODEL_PATH = "model_reduced.pkl"
ENCODER_PATH = "label_encoder.pkl"
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

# ROUTE FOR PREDICTION FORM
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get form data using the correct HTML names
        sem1_approved = max(0, int(request.form["sem1_approved"]))
        sem1_grade_10 = max(0, float(request.form["sem1_grade"]))
        sem1_eval = max(0, int(request.form["sem1_eval"]))

        sem2_approved = max(0, int(request.form["sem2_approved"]))
        sem2_grade_10 = max(0, float(request.form["sem2_grade"]))
        sem2_eval = max(0, int(request.form["sem2_eval"]))

        age = max(0, int(request.form["age"]))
        fees = int(request.form["fees"])

        # Convert grades to 20-point scale
        sem1_grade_20 = sem1_grade_10 * 2
        sem2_grade_20 = sem2_grade_10 * 2

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

    return render_template("index.html")  # show the form on GET

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
