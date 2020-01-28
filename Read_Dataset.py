# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#import Data_Processing_and_Filtering

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

dataset_new=dataset[dataset.Vehicle_ID < 201]#(Data_Processing_and_Filtering.number_of_vehicle()+1)]
dataset_new.to_csv("dataset_new.csv", index=False)

dataset_new = pd.read_csv('dataset_new.csv', delimiter=',')

# Get names of indexes for which lane has value >7
indexNames1 = dataset_new[ dataset_new['Lane_ID'] > 7 ].index
 
# Delete these row indexes from dataFrame
dataset_new.drop(indexNames1 , inplace=True)

# Get names of indexes for which vehicle has class value not 2
indexNames2 = dataset_new[ dataset_new['v_Class'] != 2 ].index
 
# Delete these row indexes from dataFrame
dataset_new.drop(indexNames2 , inplace=True)

dataset_final=dataset_new.sort_values(by=["Vehicle_ID"], ascending=True)

Veh_ID_max=max(dataset_final['Vehicle_ID'])
print("Vehicle_ID_max="+str(Veh_ID_max))

v_class=max(dataset_final['v_Class'])
print("v_class="+str(v_class))

dataset_final.to_csv("dataset_in_git.csv", index=False)
'''
df = pd.DataFrame(dataset_new,columns=['Local_X','Global_Time'])
df.plot(x ='Global_Time', y='Local_X', kind = 'scatter')
df = pd.DataFrame(dataset_new,columns=['Global_Time','Local_Y'])
df.plot(x ='Global_Time', y='Local_Y', kind = 'scatter')
df = pd.DataFrame(dataset,columns=['Frame_ID','Vehicle_ID'])
df.plot(x ='Frame_ID', y='Vehicle_ID', kind = 'scatter')
'''
