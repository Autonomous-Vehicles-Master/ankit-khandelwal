# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:10:55 2019

@author: Ankit
"""
import pandas as pd
dataset_LSTM = pd.read_csv('dataset_LSTM.csv', delimiter=',')
counter_list=[0 for i in range(40)]
m=0
x_vel=[0 for i in range(141280)]
y_vel=[0 for i in range(141280)]
for j in range(1,141280,1):
    counter_list[(dataset_LSTM['Vehicle_ID'][j]-1)]=counter_list[(dataset_LSTM['Vehicle_ID'][j]-1)]+1 
print(counter_list)
for k in range(40):
    for l in range(0,counter_list[k],1):
        if(l==0):
            x_vel[m]=0
            y_vel[m]=0
        else:
            x_vel[m]=(((dataset_LSTM['Local_X'][l]-dataset_LSTM['Local_X'][l-1]))/(1E-13))*(1E-10)
            y_vel[m]=(((dataset_LSTM['Local_Y'][l]-dataset_LSTM['Local_Y'][l-1]))/(1E-13))*(1E-15)
        m=m+1
dataset_LSTM['x_Vel']=x_vel
dataset_LSTM['y_Vel']=y_vel
dataset_LSTM.to_csv("dataset_LSTM.csv", index=False)
                     
dataset = pd.read_csv('dataset_LSTM.csv', delimiter=',', nrows=counter_list[0])
    
df = pd.DataFrame(dataset,columns=['x_Vel','Global_Time'])
df.plot(x ='Global_Time', y='x_Vel', kind = 'scatter')
df = pd.DataFrame(dataset,columns=['y_Vel','Global_Time'])
df.plot(x ='Global_Time', y='y_Vel', kind = 'scatter')    