#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:49:17 2025

@author: viaansharma
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
load_model = pickle.load(open('/Users/viaansharma/ML_COURSE/MachineLearningApps/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    #prediction 

    input_data_as_numpy_array=np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction=load_model.predict(input_data_reshaped)
    
    print(prediction)
    
    if (prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
        
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()