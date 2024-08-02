import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Load the model
model = joblib.load('notebook/strokemodel.pkl')

import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_image});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use the function to set your downloaded image as the background
set_background('path_to_image/stroke_image.jpg')

# Title of the app with custom color
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Human Stroke Prediction App</h1>", unsafe_allow_html=True)

# Display a header image
#image = Image.open('path_to_image/stroke_image.jpg')
#st.image(image, use_column_width=True)

# User input
st.header('Enter the patient details:')
st.markdown("<h3 style='color: #4682B4;'>Personal Information:</h3>", unsafe_allow_html=True)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.number_input('Age', min_value=0, max_value=120, value=25)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])

st.markdown("<h3 style='color: #4682B4;'>Lifestyle Information:</h3>", unsafe_allow_html=True)
work_type = st.selectbox('Work Type', ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Collect the input data
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Encode categorical features using LabelEncoder
label_encoders = {}
for column in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    le = LabelEncoder()
    input_data[column] = le.fit_transform(input_data[column])
    label_encoders[column] = le

# Prediction and output
if st.button('Predict', help="Click to get prediction"):
    try:
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the result with custom message and color
        if prediction[0] == 1:
            st.markdown("<h2 style='color: #FF4500;'>The patient is likely to have a stroke.</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: #32CD32;'>The patient is unlikely to have a stroke.</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.write('Error during prediction:', str(e))

# Run the app
if __name__ == '__main__':
    st.write('Streamlit web application for human stroke prediction.')