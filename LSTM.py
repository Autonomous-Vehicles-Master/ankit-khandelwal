# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:38:26 2019

@author: Ankit
"""
import numpy as np
import statistics
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
veh_to_predict=595

#'''
#data pre-processing
Data_Processing_and_Filtering.lstm_data_processing(11)
#'''

read_data=pd.read_csv('LSTM_dt_to_norm.csv', delimiter=',')  

x_hat_min=min(read_data['x_hat'])+1 #+1 because min value was min-1 to create vehicle0
print("x_hat_min="+str(x_hat_min*0.3048)) # feet to meters
y_vel_hat_min=min(read_data['y_Vel_hat'])+1 #+1 because min value was min-1 to create vehicle0
print("y_vel_hat_min="+str(y_vel_hat_min*0.3048)) # feet to meters

x_hat_max=max(read_data['x_hat'])
print("x_hat_max="+str(x_hat_max*0.3048))# feet to meters
y_vel_hat_max=max(read_data['y_Vel_hat'])
print("y_vel_hat_max="+str(y_vel_hat_max*0.3048))# feet to meters

min_nu_of_instances=150 # (Prediction time)x10 because frequency is 10 hz. unit is x100ms
#Feature_Matrix_creation.number_of_instances()

print("Prediction is done for "+str(min_nu_of_instances/10)+" Seconds")

x_hat_error=[0 for t in range(min_nu_of_instances)]
y_vel_hat_error=[0 for t in range(min_nu_of_instances)]

x_hat_error_abs=[0 for t in range(min_nu_of_instances)]
y_vel_hat_error_abs=[0 for t in range(min_nu_of_instances)]

x_hat_error_modified=[0 for t in range(min_nu_of_instances)]
y_vel_hat_error_modified=[0 for t in range(min_nu_of_instances)]

index=[t for t in range(min_nu_of_instances)]
x_ref=[t for t in range(min_nu_of_instances)]
xr=0
x_pre=[t for t in range(min_nu_of_instances)]
xp=0
v_y_ref=[t for t in range(min_nu_of_instances)]
vr=0
v_y_pre=[t for t in range(min_nu_of_instances)]
vp=0

Loss=0

# define, train and crossvalidate model 
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
    model.compile(loss= 'mean_squared_error' , optimizer= 'adam' , metrics=['mean_squared_error'])
    print(model.summary())
    
    #model=load_model('LSTM.h5')
    
    #Train Model
    for i in range(1,Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)-(cv_set_size)+1,1):
        X, Y = Feature_Matrix_creation.Feature_Matrix_creation(i)
        print(i)
        X_train = np.reshape(X, (1, X.shape[0], X.shape[1]))
        Y_train = np.reshape(Y, (1, Y.shape[0], Y.shape[1]))
        history=model.fit(X_train, Y_train, epochs=1, verbose=2, shuffle=False)
        print(history.history)
        loss_train[i-1]=history.history['mean_squared_error']
    
    model.save('LSTM.h5')
    #model=load_model('LSTM.h5')
    #Cross Validate
    k=0
    for j in range(Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)-(cv_set_size)+1,
                   Data_Processing_and_Filtering.number_of_vehicle()-(test_set_size)+1,1):
        X, Y = Feature_Matrix_creation.Feature_Matrix_creation(j)
        print(j)
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
    
    return(statistics.mean(loss_train))

'''
Loss=define_model()#0.013821252612655776
print('Average loss normalized='+str(Loss))
'''
model=load_model('LSTM.h5')

#Predict Results
X1, Y1 = Feature_Matrix_creation.Feature_Matrix_creation(veh_to_predict)
X_test = np.reshape(X1, (1, X1.shape[0], X1.shape[1]))
#Y_train = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
predictions = model.predict(X_test, verbose=2)

ds = pd.read_csv('Individual_datasets_filtered/data_'+str(veh_to_predict)+'.csv', delimiter=',', nrows=min_nu_of_instances)
ds = ds.sort_values(by=["Indexer"], ascending=True) 

# calculate errors with and without sign
for i in range(min_nu_of_instances):
    x_hat_error[i]=(predictions[0,i,0]-Y1[i,0])
    y_vel_hat_error[i]=(predictions[0,i,1]-Y1[i,1])
    x_hat_error_abs[i]=abs(predictions[0,i,0]-Y1[i,0])
    y_vel_hat_error_abs[i]=abs(predictions[0,i,1]-Y1[i,1])

#calculate average static error
x_hat_error_avg=statistics.mean(x_hat_error)
y_vel_hat_error_avg=statistics.mean(y_vel_hat_error)

# cslculate average absolute error
x_hat_error_avg_abs=statistics.mean(x_hat_error_abs)
y_vel_hat_error_avg_abs=statistics.mean(y_vel_hat_error_abs)

print('x_hat_error='+str(x_hat_error_avg_abs*(x_hat_max-x_hat_min)*0.3048))
print('y_vel_hat_error='+str(y_vel_hat_error_avg_abs*(y_vel_hat_max-y_vel_hat_min)*0.3048))

#calculate actual data and update to SI units
for num in predictions[0,:min_nu_of_instances,0]:
    #num = (num * (x_hat_max-x_hat_min))+x_hat_min
    num = ((num-x_hat_error_avg) * (x_hat_max-x_hat_min))+x_hat_min
    x_pre[xp]=num*0.3048 #feet to meters
    xp=xp+1
     
for num in predictions[0,:min_nu_of_instances,1]:
    #num = (num * (y_vel_hat_max-y_vel_hat_min))+y_vel_hat_min
    num = ((num-y_vel_hat_error_avg) * (y_vel_hat_max-y_vel_hat_min))+y_vel_hat_min
    v_y_pre[vp]=num*0.3048*10 #feet/100ms to m/s
    vp=vp+1

#update reference value unit to SI units     
for num in Y1[:min_nu_of_instances,0]:
    num = (num * (x_hat_max-x_hat_min))+x_hat_min
    x_ref[xr]=num*0.3048 #feet to meters
    xr=xr+1
     
for num in Y1[:min_nu_of_instances,1]:
    num = (num * (y_vel_hat_max-y_vel_hat_min))+y_vel_hat_min
    v_y_ref[vr]=num*0.3048*10 #feet/100ms to m/s
    vr=vr+1

#calculate modified absolute errors
for k in range(min_nu_of_instances):
    x_hat_error_modified[k]=abs(x_pre[k]-x_ref[k])
    y_vel_hat_error_modified[k]=abs(v_y_pre[k]-v_y_ref[k])

x_hat_error_modified_average=statistics.mean(x_hat_error_modified)
y_vel_hat_error_modified_average=statistics.mean(y_vel_hat_error_modified)

print('x_hat_error_modified='+str(x_hat_error_modified_average))
print('y_vel_hat_error_modified='+str(y_vel_hat_error_modified_average))

'''
plt.plot(index,predictions[0,:min_nu_of_instances,0],c='r',label='X_Target')
plt.plot(index,Y1[:min_nu_of_instances,0],label='X_Target_Prediction')
plt.xlabel('time in 100ms') 
plt.ylabel('X Position Normalized') 
plt.title('X Position Comparision')
plt.show()
#plt.legend()

plt.plot(index,predictions[0,:min_nu_of_instances,1],c='r',label='V_Y_Target')
plt.plot(index,Y1[:min_nu_of_instances,1],label='V_Y_Target_Prediction')
plt.xlabel('time in 100ms') 
plt.ylabel('Y Velocity Normalized') 
plt.title('Y Velocity Comparision')
plt.show()
#plt.legend()
'''
#'''
#plt.plot(index,l_id_pre,c='g')
plt.plot(index,x_pre,c='r')
plt.plot(index,x_ref)
plt.xticks(np.arange(0, min_nu_of_instances, 5))
plt.yticks(np.arange(0, 84*0.3048, 12*0.3048)) #to show lane markings
plt.grid(axis='y', linestyle='-')
plt.xlabel('time(x100 ms)') 
plt.ylabel('Lateral Position(m)') 
#plt.title('Lateral Position Comparision')
plt.legend(['Predicted', 'Actual'], loc='upper left')
plt.rcParams["figure.figsize"] = (20,3)
plt.show()
#plt.legend()

plt.plot(index,v_y_pre,c='r')
plt.plot(index,v_y_ref)
plt.xticks(np.arange(0, min_nu_of_instances, 5))
plt.xlabel('time(x100 ms)') 
plt.ylabel('Longitudenal Velocity(m/s)') 
#plt.title('Longitudenal Velocity Comparision')
plt.legend(['Predicted', 'Actual'], loc='upper left')
plt.rcParams["figure.figsize"] = (20,3)
plt.show()
#plt.legend()
#'''

#'''
os.remove("dataset_final.csv")
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
#'''