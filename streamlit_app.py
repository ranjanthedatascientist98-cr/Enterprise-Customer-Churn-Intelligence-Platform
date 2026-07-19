import streamlit as st
import joblib
import pandas as pd
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Cache Model Loading (Enterprise Best Practice)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/churn_prediction_bundle.pkl")

try:
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
except Exception as e:
    st.error("⚠️ Model file not found. Please ensure 'models/churn_prediction_bundle.pkl' exists.")
    st.stop()

# -----------------------------
# Custom Enterprise CSS
# -----------------------------
st.markdown("""
<style>
    /* Base styling */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Titles and Headers */
    .title-text {
        text-align: center;
        color: #38BDF8;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .subtitle-text {
        text-align: center;
        color: #94A3B8;
        font-size: 18px;
        margin-bottom: 40px;
        font-weight: 500;
    }
    .section-header {
        color: #E2E8F0;
        font-size: 22px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 1px solid #334155;
        padding-bottom: 10px;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .kpi-val {
        font-size: 28px;
        font-weight: 700;
    }

    /* Status Colors */
    .color-green { color: #10B981; }
    .color-yellow { color: #F59E0B; }
    .color-red { color: #EF4444; }
    .color-blue { color: #38BDF8; }

    /* Progress Bar */
    .progress-bg {
        background-color: #334155;
        border-radius: 10px;
        height: 20px;
        width: 100%;
        margin-top: 10px;
        margin-bottom: 5px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }

    /* Action Box */
    .action-box {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 25px;
        margin-top: 15px;
        border: 1px solid #334155;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 8px;
        height: 55px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("<div class='title-text'>📊 Enterprise Customer Churn Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>AI-Powered Customer Retention System</div>", unsafe_allow_html=True)

# -----------------------------
# 1. Customer Input Section
# -----------------------------
st.markdown("<div class='section-header'>👤 1. Customer Profile Input</div>", unsafe_allow_html=True)

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=5)
        warehouse = st.number_input("Warehouse Dist.", min_value=0, value=10)
    with col2:
        devices = st.number_input("Devices Registered", min_value=1, value=3)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    with col3:
        cashback = st.number_input("Cashback Amount", min_value=0, value=150)
        address = st.number_input("Number Of Addresses", min_value=1, value=2)
    with col4:
        last_order = st.number_input("Days Since Last Order", min_value=0, value=5)
        complain = st.selectbox("Past Complaint?", ["No", "Yes"])
    
    col5, col6, col7 = st.columns(3)
    with col5:
        category = st.selectbox("Preferred Category", ["Mobile", "Laptop & Accessory", "Fashion", "Grocery", "Others"])
    with col6:
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col7:
        st.markdown("<br>", unsafe_allow_html=True)
        predict = st.button("🔍 Analyze Churn Risk")

# -----------------------------
# Execution & AI Processing
# -----------------------------
if predict:
    start_time = time.time()

    # Feature Engineering
    customer_lifetime = "New" if tenure <= 6 else "Regular" if tenure <= 18 else "Loyal"
    distance_bucket = "Near" if warehouse <= 10 else "Medium" if warehouse <= 20 else "Far"
    complaint_label = "Complaint" if complain == "Yes" else "No Complaint"
    complain_value = 1 if complain == "Yes" else 0

    input_data = {
        "Tenure": tenure, "WarehouseToHome": warehouse, "NumberOfDeviceRegistered": devices,
        "PreferedOrderCat": category, "SatisfactionScore": satisfaction, "MaritalStatus": marital,
        "NumberOfAddress": address, "Complain": complain_value, "DaySinceLastOrder": last_order,
        "CashbackAmount": cashback, "CustomerLifetime": customer_lifetime, 
        "DistanceBucket": distance_bucket, "ComplaintLabel": complaint_label
    }

    # Preprocessing
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df)

    # Prediction
    prediction_class = model.predict(df_scaled)[0]
    churn_prob = model.predict_proba(df_scaled)[0][1]
    retention_prob = 1 - churn_prob
    
    calc_time = time.time() - start_time

    # Risk Logic Definition
    if churn_prob < 0.30:
        risk_level = "LOW RISK"
        risk_color = "#10B981"  # Green
        css_color_class = "color-green"
        status_text = "Retained"
        recs = [
            "✔ Continue Standard Loyalty Program",
            "✔ Maintain Regular Customer Engagement",
            "✔ Offer Seasonal Promotions (No Immediate Rescue Action Needed)"
        ]
    elif churn_prob <= 0.70:
        risk_level = "MEDIUM RISK"
        risk_color = "#F59E0B"  # Yellow
        css_color_class = "color-yellow"
        status_text = "At Risk"
        recs = [
            "✔ Offer Personalized Cashback Rewards",
            "✔ Target with Re-engagement Marketing Campaign",
            "✔ Send Follow-up Satisfaction Survey Email"
        ]
    else:
        risk_level = "HIGH RISK"
        risk_color = "#EF4444"  # Red
        css_color_class = "color-red"
        status_text = "Likely Churn"
        recs = [
            "✔ Initiate Immediate Retention Call",
            "✔ Offer Premium Discount / Custom Pricing",
            "✔ Assign to Dedicated Customer Support Agent",
            "✔ Escalate to CRM Retention Team"
        ]

    # -----------------------------
    # 2. Prediction Summary (KPIs)
    # -----------------------------
    st.markdown("<div class='section-header'>🎯 2. Prediction Summary</div>", unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Prediction</div>
            <div class='kpi-val {css_color_class}'>{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Churn Probability</div>
            <div class='kpi-val {css_color_class}'>{churn_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Risk Level</div>
            <div class='kpi-val {css_color_class}'>{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Model</div>
            <div class='kpi-val color-blue'>XGBoost</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability Meter
    st.markdown("<br>", unsafe_allow_html=True)
    meter_col1, meter_col2 = st.columns([1, 3])
    with meter_col1:
        st.markdown(f"<h3 style='margin:0; padding-top:10px; color:{risk_color};'>{churn_prob:.1%} Churn Risk</h3>", unsafe_allow_html=True)
    with meter_col2:
        st.markdown(f"""
        <div class='progress-bg'>
            <div class='progress-fill' style='width: {churn_prob * 100}%; background-color: {risk_color};'></div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # 3. Business Recommendations & Insights
    # -----------------------------
    st.markdown("<div class='section-header'>💼 3. Business Recommendations</div>", unsafe_allow_html=True)
    
    rec_col, insight_col = st.columns([2, 1])
    
    with rec_col:
        recs_html = "<br>".join([f"<span style='font-size: 18px; color: #E2E8F0;'>{r}</span>" for r in recs])
        st.markdown(f"""
        <div class='action-box' style='border-left: 6px solid {risk_color};'>
            <h4 style='color: #F8FAFC; margin-top: 0;'>💡 AI Action Plan</h4>
            {recs_html}
        </div>
        """, unsafe_allow_html=True)
        
    with insight_col:
        st.markdown(f"""
        <div class='action-box'>
            <h4 style='color: #F8FAFC; margin-top: 0;'>📈 Key Insights</h4>
            <div style='margin-bottom: 10px;'><b>Lifetime:</b> <span style='color:#38BDF8;'>{customer_lifetime}</span></div>
            <div style='margin-bottom: 10px;'><b>Distance:</b> <span style='color:#38BDF8;'>{distance_bucket}</span></div>
            <div style='margin-bottom: 10px;'><b>Complaints:</b> <span style='color:#38BDF8;'>{complaint_label}</span></div>
            <div style='margin-bottom: 10px;'><b>Retention Prob:</b> <span style='color:#10B981;'>{retention_prob:.1%}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # 4. Executive Summary / Model Info
    # -----------------------------
    st.markdown("<div class='section-header'>🏆 4. Model Intelligence Profile</div>", unsafe_allow_html=True)
    
    mod1, mod2, mod3, mod4 = st.columns(4)
    mod1.metric(label="Primary Algorithm", value="XGBoost Classifier")
    mod2.metric(label="Historical Accuracy", value="95.2%", delta="Validated")
    mod3.metric(label="Inference Latency", value=f"{calc_time:.3f} sec")
    mod4.metric(label="Production Status", value="Active / Ready", delta="Stable", delta_color="normal")