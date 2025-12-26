print("TRAINING STARTED")

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_dropout.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"

# Load data
df = pd.read_csv(DATA_PATH, sep=None, engine="python")

# Encode target
le = LabelEncoder()
df["Target_encoded"] = le.fit_transform(df["Target"])

X = df.drop(columns=["Target", "Target_encoded"])
y = df["Target_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and encoder
joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)

print("Model and encoder saved successfully")
