# Read CSV file and create a form to filter the data

import streamlit as st
import pandas as pd
import plotly.express as px


st.title('MLOPS Streamlit App :rocket:')
st.header('Welcome to the MLOPS Streamlit App')
st.subheader('This is a simple Streamlit app to demonstrate to read CSV file and create a form to filter the data.')

form = st.form(key='my_form')
name = form.text_input(label='Enter your name')
email = form.text_input(label='Enter your email')
age = form.slider(label='Select your age', min_value=0, max_value=100, step=1)
password = form.text_input(label='Enter your password', type='password')
message = form.text_area(label='Enter your message', max_chars = 200)
submit_button = form.form_submit_button(label='Submit')
if submit_button:
    ndf = pd.DataFrame({'Name': [name], 'Email': [email], 'Age': [age], 'Password': [password], 'Message': [message]})
    st.write(ndf)
    ndf.to_csv('user_data.csv', index=False)
    st.balloons()
