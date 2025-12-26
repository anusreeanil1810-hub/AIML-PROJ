from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_dropout.csv"

# Load data
df = pd.read_csv(DATA_PATH, sep=None, engine="python")

# Encode target
le = LabelEncoder()
df["Target_encoded"] = le.fit_transform(df["Target"])

X = df.drop(columns=["Target", "Target_encoded"])
y = df["Target_encoded"]

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# Get feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Show top features
print("\nTop 15 Important Features:\n")
print(importance_df.head(15))
