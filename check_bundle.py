import joblib

bundle = joblib.load("models/churn_prediction_bundle.pkl")

print(bundle.keys())