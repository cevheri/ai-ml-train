# Predict car price


import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import xlrd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    # data columns : Price,Make,Model,Trim,Type,Mileage,Cylinder,Liter,Doors,Cruise,Sound,Leather
    df = pd.read_excel('cars.xls')
    X = df.drop('Price', axis=1)
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Mileage', 'Cylinder', 'Liter', 'Doors', ]),  # // 'Cruise', 'Sound', 'Leather']),
            ('cat', OneHotEncoder(), ['Make', 'Model', 'Trim', 'Type'])
        ]
    )

    modelling = LinearRegression()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', modelling)])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    rmse = mean_squared_error(prediction, y_test) ** 0.5
    r2 = r2_score(prediction, y_test)

    def car_price_predictor(car_make, car_model, car_trim, car_type, car_mileage, car_cylinder, car_liter, car_doors, car_cruise, car_sound, car_leather):
        input_data = {'Make': [car_make], 'Model': [car_model], 'Trim': [car_trim], 'Type': [car_type], 'Mileage': [car_mileage],
                      'Cylinder': [car_cylinder], 'Liter': [car_liter], 'Doors': [car_doors], 'Cruise': [car_cruise], 'Sound': [car_sound], 'Leather': [car_leather]}
        input_df = pd.DataFrame(input_data)
        input_prediction = pipeline.predict(input_df)
        return input_prediction

    st.title('MLOPS Streamlit Car Price Predictor App :car:')
    st.header('Welcome to the MLOPS Streamlit Car Price Predictor App')
    st.write('This is a simple Streamlit app to demonstrate the deployment of a Machine Learning model using Streamlit to predict the car price.')

    # select unique make from the dataset
    input_make = st.selectbox('Select the Make', df['Make'].unique())
    # select unique model from the selected make
    input_model = st.selectbox('Select the Model', df[df['Make'] == input_make]['Model'].unique())
    # select unique trim from the selected make and model
    input_trim = st.selectbox('Select the Trim', df[(df['Make'] == input_make) & (df['Model'] == input_model)]['Trim'].unique())
    # select unique type from the dataset
    input_type = st.selectbox('Select the Type', df['Type'].unique())

    input_milage = st.number_input('Enter the Milage', min_value=1000, max_value=100000, step=1000)
    input_cylinder = st.slider('Select the Cylinder', min_value=0, max_value=10, step=1)
    input_liter = st.slider('Select the Liter', min_value=0, max_value=10, step=1)
    input_doors = st.slider('Select the Doors', min_value=0, max_value=10, step=1)

    input_cruise = st.radio('Select the Cruise', ('Yes', 'No'))
    input_sound = st.radio('Select the Sound', ('Yes', 'No'))
    input_leather = st.radio('Select the Leather', ('Yes', 'No'))

    submit_button = st.button('Predict Car Price')
    if submit_button:
        cruise = 1 if input_cruise == 'Yes' else 0
        sound = 1 if input_sound == 'Yes' else 0
        leather = 1 if input_leather == 'Yes' else 0
        price = car_price_predictor(car_make=input_make,
                                    car_model=input_model,
                                    car_trim=input_trim,
                                    car_type=input_type,
                                    car_mileage=input_milage,
                                    car_cylinder=input_cylinder,
                                    car_liter=input_liter,
                                    car_doors=input_doors,
                                    car_cruise=cruise,
                                    car_sound=sound,
                                    car_leather=leather)
        st.write('Car Price should be $ {} from Machine Learning'.format(round(price[0], 2)))
        st.balloons()


if __name__ == '__main__':
    main()

# Run the app using the command streamlit run app_model_pickle.py
# freeze the requirements using the command pip freeze > requirements.txt
# streamlit run app.py
