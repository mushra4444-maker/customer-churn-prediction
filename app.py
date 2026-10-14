"""Streamlit interface for the customer churn prediction model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction")
st.write("Estimate whether a customer may leave based on service and account details.")
st.caption("Demo project using a synthetic dataset. Predictions are educational and should not be treated as business certainty.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` first.")
    st.stop()
model = joblib.load(MODEL_PATH)

with st.form("customer_details"):
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        support = st.selectbox("Tech support", ["Yes", "No"])
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    with col2:
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
        total = st.number_input("Total charges", min_value=0.0, max_value=25000.0, value=float(tenure * monthly), step=10.0)
    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = pd.DataFrame([{
        "tenure_months": tenure,
        "phone_service": phone,
        "internet_service": internet,
        "tech_support": support,
        "contract_type": contract,
        "paperless_billing": paperless,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
    }])
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(probability >= 0.5)
    st.subheader("Prediction result")
    st.metric("Estimated churn probability", f"{probability:.1%}")
    if prediction:
        st.error("Higher churn risk: this model predicts the customer may leave.")
        st.write("Possible action: review service issues, offer support, or consider a suitable retention offer.")
    else:
        st.success("Lower churn risk: this model predicts the customer is less likely to leave.")
    st.progress(probability)
