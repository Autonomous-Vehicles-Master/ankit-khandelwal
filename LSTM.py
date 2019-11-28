# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:38:26 2019

@author: Ankit
"""
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot as plt
import pandas as pd
import os
import Data_Processing_and_Filtering
import Feature_Matrix_creation

test_set_size=10
cv_set_size=50
veh_to_predict=194

Data_Processing_and_Filtering.lstm_data_processing(11)

read_data=pd.read_csv('LSTM_dts.csv', delimiter=',')  

x_hat_min=min(read_data['x_hat'])
print("x_hat_min="+str(x_hat_min))
y_vel_hat_min=min(read_data['y_Vel_hat'])
print("y_vel_hat_min="+str(y_vel_hat_min))
v_type_min=min(read_data['v_Type'])
print("v_type_min="+str(v_type_min))

x_hat_max=max(read_data['x_hat'])
print("x_hat_max="+str(x_hat_max))
y_vel_hat_max=max(read_data['y_Vel_hat'])
print("y_vel_hat_max="+str(y_vel_hat_max))
v_type_max=max(read_data['v_Type'])
print("v_type_max="+str(v_type_max))

min_nu_of_instances=Feature_Matrix_creation.number_of_instances()
index=[t for t in range(min_nu_of_instances)]
x_ref=[t for t in range(min_nu_of_instances)]
xr=0
x_pre=[t for t in range(min_nu_of_instances)]
xp=0
v_y_ref=[t for t in range(min_nu_of_instances)]
vr=0
v_y_pre=[t for t in range(min_nu_of_instances)]
vp=0
 
def define_model():
    X, Y = Feature_Matrix_creation.Feature_Matrix_creation(1)
    loss_train = [0 for t in range(Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)-(cv_set_size))]
    loss_cv = [0 for t in range(cv_set_size)]
    indexer_train=[t for t in range(Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)-(cv_set_size))]
    indexer_cv = [t for t in range(cv_set_size)]
    # define model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[0], X.shape[1]),return_sequences=True))
    model.add(Dense(256, activation= 'linear' ))
    model.add(Dense(128, activation= 'linear' ))
    model.add(Dense(2, activation= 'linear' ))
    model.compile(loss= 'mean_squared_error' , optimizer= 'adam' )
    print(model.summary())
    
    #Train Model
    for i in range(1,Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)-(cv_set_size)+1,1):
        X, Y = Feature_Matrix_creation.Feature_Matrix_creation(i)
        X_train = np.reshape(X, (1, X.shape[0], X.shape[1]))
        Y_train = np.reshape(Y, (1, Y.shape[0], Y.shape[1]))
        model.fit(X_train, Y_train, epochs=1, verbose=2, shuffle=False)
        loss_train[i-1]= model.evaluate(X_train, Y_train, verbose=2)
    
    model.save('LSTM.h5')
    
    #Cross Validate
    k=0
    for j in range(Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)-(cv_set_size)+1,
                   Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)+1,1):
        X, Y = Feature_Matrix_creation.Feature_Matrix_creation(j)
        X_train = np.reshape(X, (1, X.shape[0], X.shape[1]))
        Y_train = np.reshape(Y, (1, Y.shape[0], Y.shape[1]))
        loss_cv[k]= model.evaluate(X_train, Y_train, verbose=2)
        k=k+1
        
    # Plot training & validation loss values
    plt.plot(indexer_train,loss_train)
    plt.plot(indexer_cv,loss_cv, c='r')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
        


define_model()

model=load_model('LSTM.h5')

#Predict Results
X1, Y1 = Feature_Matrix_creation.Feature_Matrix_creation(veh_to_predict)
X_test = np.reshape(X1, (1, X1.shape[0], X1.shape[1]))
#Y_train = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
predictions = model.predict(X_test, verbose=2)

for num in predictions[0,:,0]:
     num = (num * (x_hat_max-x_hat_min))+x_hat_min
     x_pre[xp]=num
     xp=xp+1
     
for num in predictions[0,:,1]:
     num = (num * (y_vel_hat_max-y_vel_hat_min))+y_vel_hat_min
     v_y_pre[vp]=num
     vp=vp+1
     
for num in Y1[:,0]:
     num = (num * (x_hat_max-x_hat_min))+x_hat_min
     x_ref[xr]=num
     xr=xr+1
     
for num in Y1[:,1]:
     num = (num * (y_vel_hat_max-y_vel_hat_min))+y_vel_hat_min
     v_y_ref[vr]=num
     vr=vr+1
'''
plt.plot(index,predictions[0,:,0],c='r',label='X_Target')
plt.plot(index,Y1[:,0],label='X_Target_Prediction')
plt.xlabel('time in ms') 
plt.ylabel('X Position Normalized') 
plt.title('X Position Comparision')
plt.show()
#plt.legend()

plt.plot(index,predictions[0,:,1],c='r',label='V_Y_Target')
plt.plot(index,Y1[:,1],label='V_Y_Target_Prediction')
plt.xlabel('time in ms') 
plt.ylabel('Y Velocity Normalized') 
plt.title('Y Velocity Comparision')
plt.show()
#plt.legend()
'''

plt.plot(index,x_pre,c='r',label='X_Target')
plt.plot(index,x_ref,label='X_Target_Prediction')
plt.xlabel('time in ms') 
plt.ylabel('X Position Normalized') 
plt.title('X Position Comparision')
plt.show()
#plt.legend()

plt.plot(index,v_y_pre,c='r',label='V_Y_Target')
plt.plot(index,v_y_ref,label='V_Y_Target_Prediction')
plt.xlabel('time in ms') 
plt.ylabel('Y Velocity Normalized') 
plt.title('Y Velocity Comparision')
plt.show()
#plt.legend()

os.remove("dataset_final.csv")
os.remove("dataset_LSTM.csv")
os.remove("dataset_useful.csv")
os.remove("LSTM_dts.csv")
os.remove("LSTM_dt_to_norm.csv")
os.remove("LSTM_Normalized.csv")

for dl0 in range(1,Data_Processing_and_Filtering.number_of_vehicle()+1,1):
    os.remove("Individual_datasets/data_"+str(dl0)+".csv")
    os.remove("Individual_datasets_updated/data_"+str(dl0)+".csv")
    
for dl1 in range(0,Data_Processing_and_Filtering.number_of_vehicle()+1,1):
    os.remove("Individual_datasets_filtered/data_"+str(dl1)+".csv")
    os.remove("Individual_datasets_normalized/data_"+str(dl1)+".csv")

os.rmdir("Individual_datasets")
os.rmdir("Individual_datasets_filtered")
os.rmdir("Individual_datasets_updated")
os.rmdir("Individual_datasets_normalized")
