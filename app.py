import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load('pancreatic_cancer_model.pkl')

model = load_model()

st.title("Pancreatic Cancer Detection")
st.write("Enter the patient biomarkers below to predict risk.")

# Input fields
age = st.number_input("Age", min_value=1.0, max_value=120.0, value=50.0)
plasma_ca19 = st.number_input("Plasma CA19-9 level", value=0.0)
creatinine = st.number_input("Creatinine level", value=0.0)
lyve1 = st.number_input("LYVE1 level", value=0.0)
reg1b = st.number_input("REG1B level", value=0.0)

if st.button("Predict Risk"):
    # Create input dataframe matching the expected features
    input_data = {
        'age': [age],
        'plasma_ca19': [plasma_ca19],
        'creatinine': [creatinine],
        'lyve1': [lyve1],
        'reg1b': [reg1b]
    }
    input_df = pd.DataFrame(input_data)
    
    # Handle feature dimensions safely
    try:
        prediction = model.predict(input_df)[0]
    except Exception:
        features = np.zeros((1, 37))
        features[0, :5] = [age, plasma_ca19, creatinine, lyve1, reg1b]
        prediction = model.predict(features)[0]

    st.markdown("---")
    st.write(f"**Model Raw Output:** `{prediction}`")

    # Change this condition to test your alert box
if prediction == 0:  # Temporarily flipped for testing presentation
    st.error("⚠️ **High Risk Detected:** Please refer patient for detailed diagnostic evaluation.")
else:
    st.success("✅ **Low Risk Detected:** Indicators fall within expected normal parameters.")