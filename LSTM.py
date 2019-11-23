# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:38:26 2019

@author: Ankit
"""
'''
from random import random
from matplotlib import pyplot
from matplotlib.patches import PathPatch
from matplotlib.path import Path
'''
#import tensorflow as tf
#from tensorflow.keras.layers import Input, Dense, Sequential, LSTM
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import Feature_Matrix_creation
import Data_Processing_and_Filtering

#Data_Processing_and_Filtering.lstm_data_processing(11) 
X, Y = Feature_Matrix_creation.Feature_Matrix_creation(1)

# define model
model = Sequential()
model.add(LSTM(256, input_shape=(1, X.shape[1]),return_sequences=True))
model.add(Dense(256, activation= 'linear' ))
model.add(Dense(128, activation= 'linear' ))
model.add(Dense(2, activation= 'linear' ))
model.compile(loss= 'mean_squared_error' , optimizer= 'sgd' )
print(model.summary())

for i in range(1,Data_Processing_and_Filtering.no_of_veh_used-50,1):
    X, Y = Feature_Matrix_creation.Feature_Matrix_creation(i)
    X_train = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    Y_train = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
    model.fit(X_train, Y_train, epochs=1, verbose=2, shuffle=False)
    
    
for j in range(Data_Processing_and_Filtering.no_of_veh_used-50,
               Data_Processing_and_Filtering.no_of_veh_used-25,1):
    X, Y = Feature_Matrix_creation.Feature_Matrix_creation(i)
    X_CV = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    Y_CV = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
    model.evaluate(X_CV, Y_CV, verbose=0)

for k in range(Data_Processing_and_Filtering.no_of_veh_used-25,
               Data_Processing_and_Filtering.no_of_veh_used+1,1):
    X, Y = Feature_Matrix_creation.Feature_Matrix_creation(i)
    X_test = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    #Y_train = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
    predictions = model.predict(X_test, verbose=0)
