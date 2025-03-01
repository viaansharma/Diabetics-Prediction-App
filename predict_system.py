# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

#load the saved model
load_model=pickle.load(open('/Users/viaansharma/ML_COURSE/MachineLearningApps/trained_model.sav','rb'))

#prediction 
input_data=(1,103,30,38,83,43.3,0.183,33)

input_data_as_numpy_array=np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction=load_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
   