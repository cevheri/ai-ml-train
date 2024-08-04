# pip install streamlit
# Run the app using the following command
# streamlit run app.py
import pandas as pd
import streamlit as st
import plotly.express as px

st.title('MLOPS Streamlit App :rocket:')
st.header('Welcome to the MLOPS Streamlit App')
st.subheader('This is a simple Streamlit app to demonstrate the deployment of a Machine Learning model using Streamlit.')
st.write('This is a simple Streamlit app to demonstrate the deployment of a Machine Learning model using Streamlit.')
form = st.form(key='my_form')
name = form.text_input(label='Enter your name')
submit_button = form.form_submit_button(label='Submit')
if submit_button:
    st.write(f'Hello, {name}! Welcome to the MLOPS Streamlit App :rocket:')
    st.balloons()


menu= ['Home', 'About', 'Contact']
st.sidebar.selectbox('Menu', menu)


st.slider('Select a value', min_value=0, max_value=100, step=1)

st.audio('https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3')


df = pd.read_csv('prog_languages_data.csv')
figure = px.pie(df, values='Sum')
st.plotly_chart(figure)


fig2= px.bar(df, x='lang', y='Sum')
st.plotly_chart(fig2)

file = st.file_uploader('Upload a file', type=['csv'])

st.divider()
st.video('secret_of_success.mp4')
st.divider()


st.camera_input('camera_input')

st.divider()
st.date_input('date_input')
st.divider()
st.time_input('time_input')
st.divider()
st.text_input('password_input', type='password')
st.divider()
st.text_area('text_area', max_chars=200)
st.divider()
st.number_input('number_input', min_value=0, max_value=100, step=1)
st.divider()
st.radio('Gender(Select one option)', ("Single", "Married"))
st.divider()
st.checkbox('I accept the terms and conditions', value=False)
st.divider()
st.selectbox('Select a country', ('India', 'USA', 'UK'))
st.divider()
st.multiselect('Select multiple options', ('Java', 'Python', 'C++'))
st.divider()
st.image('https://www.streamlit.io/images/brand/streamlit-mark-color.png')

st.divider()
st.code('print("Hello, World!")')
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
st.dataframe(df)


st.success('This is a success message')
st.info('This is an info message')
st.warning('This is a warning message')
st.error('This is an error message')



st.divider()
#no video with supported format and mime type found
st.video('http://192.168.1.9:8080/video', format='video/mp4')
st.divider()