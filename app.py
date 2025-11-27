import streamlit as st 
import numpy as np
import pandas as pd
import joblib 

#Lets load the joblib instances  over here
with open('pipeline.joblib','rb') as file:
    preprocess = joblib.load(file)
    
with open('model.joblib','rb') as file:
    model = joblib.load(file)
    
    
    
#Lets take input for the users 
st.title('HELP-NGO ORGANIZATION')
st.subheader('This application help to identiy the develpoment of different country on the basis of various parameters which are clustered through KMeans commetig on Socio-Economic Factors.')


#Lets take the input 
gdpp = st.number_input('Enter the GPPP of a country (GDP per population)')
income = st.number_input('Enter income per population')
imports = st.number_input('Imports of goods and services per capita. Given as age of the GDP per capita')
exports = st.number_input('Exports of goods and services per capita. Given as age of the GDP per capita')
inflation = st.number_input('The measurement of the annual growth rate of the Total GDP')
life_expcy = st.number_input('The average number of years a new born child would live if the current mortality patterns are to remain the same')
fert = st.number_input('The number of children that would be born to each woman if the current age-fertility rates remain the same.')
health = st.number_input('Total health spending per capita. Given as age of GDP per capita')
child_mort = st.number_input('Death of children under 5 years of age per 1000 live births')

input_list = [child_mort,exports,health,imports,income,inflation,life_expcy,fert,gdpp]

final_input_list = preprocess.transform([input_list])

if st.button('Predict'):
    prediction = model.predict(final_input_list)[0]
    if prediction == 0:
        st.success('Developing')
    elif prediction == 2:
        st.success('Develpoed')
    else:
        st.error('UnderDeveloping')