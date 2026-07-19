import pandas as pd

from app.model_loader import model, scaler, feature_columns


def predict_customer(customer_data):
    # Convert request to dictionary
    data = customer_data.model_dump()

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    # Customer Lifetime
    if data["Tenure"] <= 6:
        data["CustomerLifetime"] = "New"
    elif data["Tenure"] <= 18:
        data["CustomerLifetime"] = "Regular"
    else:
        data["CustomerLifetime"] = "Loyal"

    # Distance Bucket
    if data["WarehouseToHome"] <= 10:
        data["DistanceBucket"] = "Near"
    elif data["WarehouseToHome"] <= 20:
        data["DistanceBucket"] = "Medium"
    else:
        data["DistanceBucket"] = "Far"

    # Complaint Label
    if data["Complain"] == 1:
        data["ComplaintLabel"] = "Complaint"
    else:
        data["ComplaintLabel"] = "No Complaint"

    # -----------------------------
    # Convert to DataFrame
    # -----------------------------
    df = pd.DataFrame([data])

    # One-Hot Encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale Features
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = int(model.predict(df_scaled)[0])

    # Probability
    probability = float(model.predict_proba(df_scaled)[0][1])

    # -----------------------------
    # Business Interpretation
    # -----------------------------
    if prediction == 1:
        prediction_label = "⚠ Customer Likely to Churn"
    else:
        prediction_label = "✅ Customer Likely to Stay"

    if probability < 0.30:
        risk_level = "Low"
        recommendation = (
            "Customer is healthy. Continue regular engagement and loyalty programs."
        )
    elif probability < 0.70:
        risk_level = "Medium"
        recommendation = (
            "Customer has moderate churn risk. Offer cashback or personalized promotions."
        )
    else:
        risk_level = "High"
        recommendation = (
            "Customer is at high risk. Immediate retention campaign is recommended."
        )

    # -----------------------------
    # API Response
    # -----------------------------
    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "churn_probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendation": recommendation
    }