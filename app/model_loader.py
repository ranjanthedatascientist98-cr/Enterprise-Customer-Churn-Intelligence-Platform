import joblib

from pathlib import Path

# Get the project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Model file location
MODEL_PATH = BASE_DIR / "models" / "churn_prediction_bundle.pkl"

# Load deployment bundle
bundle = joblib.load(MODEL_PATH)

# Extract components
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
