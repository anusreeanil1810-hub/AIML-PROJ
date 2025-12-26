from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_dropout.csv"
MODEL_PATH = BASE_DIR / "model_reduced.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH, sep=None, engine="python")

# Selected features
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

X = df[FEATURES]
y = df["Target"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nReduced Feature Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and encoder
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, BASE_DIR / "label_encoder.pkl")

print("\nModel and label encoder saved successfully.")
