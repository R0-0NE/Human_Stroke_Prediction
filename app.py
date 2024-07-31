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


# List the exact column names expected by your model after one-hot encoding
expected_columns = [
    'gender_Female', 'gender_Male', 'gender_Other',
    'ever_married_No', 'ever_married_Yes',
    'work_type_children', 'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes', 'smoking_status_Unknown',
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'
]

# Create an empty DataFrame with these columns
input_data_aligned = pd.DataFrame(columns=expected_columns)

# Assuming input_data is your input DataFrame from the Streamlit form, you need to one-hot encode it
# Replace `input_data` with the actual input DataFrame from your app
input_data_encoded = pd.get_dummies(input_data)

# Align the columns of input_data_encoded with the expected_columns
pd.set_option('future.no_silent_downcasting', True)
input_data_aligned = input_data_aligned.combine_first(input_data_encoded).fillna(0).infer_objects()

# Ensure that the columns in input_data_aligned and input_data_encoded have matching data types
for column in input_data_aligned.columns:
    if column in input_data_encoded.columns:
        input_data_aligned[column] = input_data_encoded[column]

# Prediction and output
if st.button('Predict'):
    try:
        # Make the prediction
        prediction = model.predict(input_data_aligned)
        prediction_proba = model.predict_proba(input_data_aligned)
        
        # Display the result
        if prediction[0] == 1:
            st.write('The patient is likely to have a stroke.')
            st.write(f'Probability of stroke: {prediction_proba[0][1]:.2f}')
        else:
            st.write('The patient is unlikely to have a stroke.')
            st.write(f'Probability of not having a stroke: {prediction_proba[0][0]:.2f}')
    except Exception as e:
        st.write('Error during prediction:', str(e))
# Run the app
if __name__ == '__main__':
    st.write('Streamlit web application for human stroke prediction.')