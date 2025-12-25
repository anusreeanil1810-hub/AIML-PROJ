print("MAIN FILE STARTED")

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_dropout.csv"

df = pd.read_csv(DATA_PATH, sep=None, engine="python")

print("\nDataset loaded successfully")

# -------------------------------
# STEP 1: Encode target column
# -------------------------------
le = LabelEncoder()
df["Target_encoded"] = le.fit_transform(df["Target"])

# Drop original target
X = df.drop(columns=["Target", "Target_encoded"])
y = df["Target_encoded"]

print("\nTarget encoding:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------------
# STEP 2: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------------------------------
# STEP 3: Train model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# STEP 4: Evaluate model
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
