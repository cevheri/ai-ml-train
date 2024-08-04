import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

st.title('MLOPS Streamlit App :rocket:')
st.header('Welcome to the MLOPS Streamlit App from a Pickle file')
st.subheader('This is a simple Streamlit app to demonstrate the deployment of a Machine Learning model using Streamlit from a Pickle file.')

model = pickle.load(open('salary-calculater.pkl', 'rb'))
experience = st.slider('Select your experience', min_value=0, max_value=10, step=1)
exam_score = st.slider('Select your exam score', min_value=0, max_value=10, step=1)
interview_score = st.slider('Select your interview score', min_value=0, max_value=10, step=1)
submit_button = st.button('Predict Salary')
if submit_button:
    prediction = model.predict([[experience, exam_score, interview_score]])
    st.write('Employee Salary should be $ {} from Machine Learning'.format(round(prediction[0],2)))