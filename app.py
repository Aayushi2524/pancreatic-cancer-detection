import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load trained model
model = joblib.load('pancreatic_cancer_model.pkl')

st.title("🩺 Pancreatic Cancer Early Detection System")
st.write("Enter patient clinical values below to check risk levels.")

# Input forms
st.subheader("Patient Clinical Inputs")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
plasma_ca19 = st.number_input("Plasma CA19-9 level", min_value=0.0, value=25.0)
creatinine = st.number_input("Creatinine level", min_value=0.0, value=1.0)
lyve1 = st.number_input("LYVE1 level", min_value=0.0, value=2.0)
reg1b = st.number_input("REG1B level", min_value=0.0, value=3.0)

# Prediction Logic
if st.button("Predict Risk"):
    if hasattr(model, "feature_names_in_"):
        cols = model.feature_names_in_
        input_df = pd.DataFrame(0, index=[0], columns=cols)
        
        for col in cols:
            col_lower = col.lower()
            if 'age' in col_lower:
                input_df[col] = age
            elif 'ca19' in col_lower or 'plasma' in col_lower:
                input_df[col] = plasma_ca19
            elif 'creatinine' in col_lower:
                input_df[col] = creatinine
            elif 'lyve1' in col_lower:
                input_df[col] = lyve1
            elif 'reg1b' in col_lower:
                input_df[col] = reg1b
                
       try:
        prediction = model.predict(input_df)[0]
    except Exception:
        features = np.zeros((1, 37))
        features[0, :5] = [age, plasma_ca19, creatinine, lyve1, reg1b]
        prediction = model.predict(features)[0]

    st.markdown("---")
    st.write(f"**Model Raw Output:** `{prediction}`")

    if prediction != 0:
        st.error("⚠️ **High Risk Detected:** Please refer patient for detailed diagnostic evaluation.")
    else:
        st.success("✅ **Low Risk Detected:** Indicators fall within expected normal parameters.")