import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import os

# Load the model
model_path = 'model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Model file not found. Please ensure the model is available in the correct path.")
    st.stop()

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)  
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)  


st.title("salary Prediction")
st.write("This app predicts whether a customer will leave the bank based on their information.")

# User inputs
st.sidebar.header("Input Features")

credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Convert categorical inputs
gender_encoded = label_encoder_gender.transform([gender])[0]
geography_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()

# Combine inputs into a single array
input_data = np.array([[credit_score, gender_encoded, age, tenure, balance, num_of_products,
                        1 if has_cr_card == "Yes" else 0, 1 if is_active_member == "Yes" else 0,
                        estimated_salary]])
input_data = np.concatenate([input_data, geography_encoded], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write("### Prediction")
st.write(f"The predicted salary is: ${predicted_salary:,.2f}")