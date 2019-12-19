# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import Data_Processing_and_Filtering

dataset = pd.read_csv('Dataset.csv', delimiter=',')
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

dataset_new=dataset[dataset.Vehicle_ID < (Data_Processing_and_Filtering.number_of_vehicle()+1)]
dataset_new.to_csv("dataset_new.csv", index=False)

dataset_new = pd.read_csv('dataset_new.csv', delimiter=',')

# Get names of indexes for which column Age has value 30
indexNames = dataset_new[ dataset_new['Lane_ID'] > 7 ].index
 
# Delete these row indexes from dataFrame
dataset_new.drop(indexNames , inplace=True)

dataset_final=dataset_new.sort_values(by=["Vehicle_ID"], ascending=True)

Lane_ID_max=max(dataset_final['Lane_ID'])
print("Lane_ID_max="+str(Lane_ID_max))

dataset_final.to_csv("dataset_in_git.csv", index=False)
'''
df = pd.DataFrame(dataset_new,columns=['Local_X','Global_Time'])
df.plot(x ='Global_Time', y='Local_X', kind = 'scatter')
df = pd.DataFrame(dataset_new,columns=['Global_Time','Local_Y'])
df.plot(x ='Global_Time', y='Local_Y', kind = 'scatter')
df = pd.DataFrame(dataset,columns=['Frame_ID','Vehicle_ID'])
df.plot(x ='Frame_ID', y='Vehicle_ID', kind = 'scatter')
'''
