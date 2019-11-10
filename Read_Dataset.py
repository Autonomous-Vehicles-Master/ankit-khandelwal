# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
dataset = pd.read_csv('dataset.csv', delimiter=',')
"""
The data set has following columns:
Vehicle_ID
Frame_ID
Total_Frames
Global_Time
Local_X
Local_Y
Global_X
Global_Y
v_length
v_Width
v_Class
v_Vel
v_Acc
Lane_ID
O_Zone
D_Zone
Int_ID
Section_ID
Direction
Movement
Preceding
Following
Space_Headway
Time_Headway
Location	
"""

dataset_new=dataset[dataset.Vehicle_ID < 41]
dataset_new.to_csv("dataset_new.csv", index=False)

dataset_new = pd.read_csv('dataset_new.csv', delimiter=',')

dataset_final=dataset_new.sort_values(by=["Vehicle_ID"], ascending=True)
dataset_final.to_csv("dataset_final.csv", index=False)
'''
df = pd.DataFrame(dataset_new,columns=['Local_X','Global_Time'])
df.plot(x ='Global_Time', y='Local_X', kind = 'scatter')
df = pd.DataFrame(dataset_new,columns=['Global_Time','Local_Y'])
df.plot(x ='Global_Time', y='Local_Y', kind = 'scatter')
df = pd.DataFrame(dataset,columns=['Frame_ID','Vehicle_ID'])
df.plot(x ='Frame_ID', y='Vehicle_ID', kind = 'scatter')
'''
