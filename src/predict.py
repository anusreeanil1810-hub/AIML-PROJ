from pathlib import Path
import joblib
import pandas as pd

print("STUDENT DROPOUT PREDICTION SYSTEM\n")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_reduced.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# Load model and encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Features (must match training order)
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

# Input collection
input_data = []

print("Enter student details:\n")
print("INPUT GUIDELINES:")
print("- Approved units & evaluations: enter non-negative integers (e.g., 0, 1, 5)")
print("- Grades: enter values between 0 and 20")
print("- Age at enrollment: enter age in years (e.g., 18, 19, 22)")
print("- Tuition fees up to date: 1 = Yes, 0 = No\n")


for feature in FEATURES:
    while True:
        try:
            value = float(input(f"{feature}: "))
            input_data.append(value)
            break
        except ValueError:
            print("Please enter a numeric value.")

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=FEATURES)

# Predict
prediction_encoded = model.predict(input_df)[0]
prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

print("\nPREDICTION RESULT:")
print("Student Status:", prediction_label)
