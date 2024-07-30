import streamlit as st
import joblib, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('notebook\strokemodel.pkl')

# Title of the app
st.title('Human Stroke Prediction App')


# User input
st.header('Enter the patient details:')
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.number_input('Age', min_value=0, max_value=120, value=25)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Convert inputs to numerical values
#gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
#ever_married_map = {'No': 0, 'Yes': 1}
#work_type_map = {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4}
#residence_type_map = {'Rural': 0, 'Urban': 1}
#smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

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


# Manually encode categorical variables
input_data_encoded = pd.get_dummies(input_data)

# Assume `feature_columns` is the list of feature columns the model was trained on
feature_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Female', 'gender_Male', 'gender_Other',
    'ever_married_No', 'ever_married_Yes',
    'work_type_children', 'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes', 'smoking_status_Unknown'
]

# Create a DataFrame with the feature columns, filled with zeros
input_data_aligned = pd.DataFrame(0, index=np.arange(1), columns=feature_columns)

# Update with the actual input data
input_data_aligned.update(input_data_encoded)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data_aligned)
    if prediction[0] == 1:
        st.write('The patient is likely to have a stroke.')
    else:
        st.write('The patient is unlikely to have a stroke.')

# Run the app
if __name__ == '__main__':
    st.write('Streamlit web application for human stroke prediction.')