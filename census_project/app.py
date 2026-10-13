import sys
import os
import streamlit as st
import pandas as pd
import joblib
from ml_utils import validate_incoming_data
# 1. Page Configuration
st.set_page_config(page_title="Donor Prediction App", page_icon="🎯", layout="centered")
st.title("🎯 Charity Donor Prediction System")
st.markdown("Enter the client's details below to predict if they are likely to donate.")

# 2. Load the Model and Business Threshold (Cached for performance)
@st.cache_resource
def load_model():
    package = joblib.load('robust_donors_model_package.pkl')
    return package['model_pipeline'], package['business_threshold']

model, threshold = load_model()

# 3. User Input Interface
st.header("Client Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=100, value=30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov'])
    education_level = st.selectbox("Education Level", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])

with col2:
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    capital_gain = st.number_input("Capital Gain ($)", min_value=0, value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=0, max_value=99, value=40)
    sex = st.radio("Gender", ['Male', 'Female'])

# 4. Prediction Button
if st.button("🔮 Predict Donation Probability"):
    
    # Aggregate raw input into a DataFrame
    raw_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education_level': [education_level],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'capital-gain': [capital_gain],
        'hours-per-week': [hours_per_week],
        'sex': [sex],
        'capital-loss': [0],
        'education-num': [10],
        'relationship': ['Husband'],
        'race': ['White'],
        'native-country': ['United-States']
    })

    # ==========================================
    # Apply Validation before Model Inference
    # ==========================================
    validated_data = validate_incoming_data(raw_data)

    st.markdown("---")
    
    # Check if the data was rejected (empty DataFrame returned)
    if validated_data.empty:
        st.error("❌ Validation Error: The input data violates the business rules (e.g., Age must be between 17 and 90).")
    
    # Proceed with prediction if data is valid
    else:
        # Calculate probability
        proba = model.predict_proba(validated_data)[0, 1]
        
        # Apply the business threshold
        if proba >= threshold:
            st.success(f"✅ **TARGET (CALL)**: This client is highly likely to donate!")
            st.balloons()
        else:
            st.error(f"❌ **IGNORE**: This client is unlikely to donate.")
        
        st.info(f"📊 **Model Probability:** {proba * 100:.1f}% (Threshold: {threshold * 100:.1f}%)")
