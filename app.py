import streamlit as st
import pickle
import numpy as np

# Set up web page title and icon
st.set_page_config(page_title="Pancreatic Cancer Detection", page_icon="🩺", layout="centered")

st.title("🩺 Pancreatic Cancer Early Detection System")
st.write("Enter patient clinical values below to check risk levels.")

# Load your saved trained model
@st.cache_resource
def load_model():
    with open('pancreatic_cancer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Create input forms for patient metrics
st.subheader("Patient Clinical Inputs")

# NOTE: Modify these input fields to match the EXACT features/columns your model was trained on
age = st.number_input("Age", min_value=1, max_value=120, value=50)
plasma_ca19 = st.number_input("Plasma CA19-9 level", min_value=0.0, value=25.0)
creatinine = st.number_input("Creatinine level", min_value=0.0, value=1.0)
lyve1 = st.number_input("LYVE1 level", min_value=0.0, value=2.0)
reg1b = st.number_input("REG1B level", min_value=0.0, value=3.0)

# Prediction Logic
if st.button("Predict Risk"):
    # Array format matching model features
    features = np.array([[age, plasma_ca19, creatinine, lyve1, reg1b]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ **High Risk Detected:** Please refer patient for detailed diagnostic evaluation.")
    else:
        st.success("✅ **Low Risk Detected:** Indicators fall within expected normal parameters.")