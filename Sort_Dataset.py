# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:08:52 2019

@author: Ankit
"""


import pandas as pd
dataset_final1 = pd.read_csv('dataset_final1.csv', delimiter=',')
"""
The data set has following columns:
"""
colnames=['Vehicle_ID',
'Frame_ID',
'Total_Frames',
'Global_Time',
'Local_X',
'Local_Y',
'Global_X',
'Global_Y',
'v_length',
'v_Width',
'v_Class',
'v_Vel',
'v_Acc',
'Lane_ID',
'O_Zone',
'D_Zone',
'Int_ID',
'Section_ID',
'Direction',
'Movement',
'Preceding',
'Following',
'Space_Headway',
'Time_Headway',
'Location']
'''
df = pd.DataFrame(dataset_new,columns=['Local_X','Global_Time'])
df.plot(x ='Global_Time', y='Local_X', kind = 'scatter')
df = pd.DataFrame(dataset_new,columns=['Global_Time','Local_Y'])
df.plot(x ='Global_Time', y='Local_Y', kind = 'scatter')
dataset_final=dataset_new.sort_values(by=["Vehicle_ID"], ascending=True)
dataset_final.to_csv("dataset_final.csv", index=False)
'''
counter_list=[0 for i in range(40)]
print(counter_list) 
print(dataset_final1['Frame_ID'][0])

dataset=[]
ds1=[]
ds2=[] 
ds3=[] 
ds4=[]
ds5=[]
ds6=[]
ds7=[]
ds8=[]
ds9=[]
ds10=[]
ds11=[]
ds12=[]
ds13=[]
ds14=[]
ds15=[]
ds16=[]
ds17=[]
ds18=[]
ds19=[]
ds20=[]
ds21=[]
ds22=[]
ds23=[]
ds24=[]
ds25=[]
ds26=[]
ds27=[]
ds28=[]
ds29=[]
ds30=[]
ds31=[]
ds32=[]
ds33=[]
ds34=[]
ds35=[]
ds36=[]
ds37=[]
ds38=[]
ds39=[]
ds40=[]
dataset_final=dataset_final1.sort_values(by=["Vehicle_ID"], ascending=True)
dataset_final.to_csv("dataset_final.csv", index=False)
for j in range(1,141281,1):
    counter_list[(dataset_final['Vehicle_ID'][j]-1)]=counter_list[(dataset_final['Vehicle_ID'][j]-1)]+1 
print(counter_list)
counter=counter_list
dst= pd.read_csv('dataset_final.csv', delimiter=',',nrows=counter_list[0])
datasetlist=[ds1,ds2,ds3,ds4,
             ds5,ds6,ds7,ds8,
             ds9,ds10,ds11,ds12,
             ds13,ds14,ds15,ds16,
             ds17,ds18,ds19,ds20,
             ds21,ds22,ds23,ds24,
             ds25,ds26,ds27,ds28,
             ds29,ds30,ds31,ds32,
             ds33,ds34,ds35,ds36,
             ds37,ds38,ds39,ds40,
             ]
ds1=dst.sort_values(by=["Local_X"], ascending=True)

ds1.to_csv("data_1.csv", index=False)

for k1 in range(39):
    counter[k1+1]=counter[k1+1]+counter[k1]
    datasetlist[k1]=0
    datasetlist[k1]= pd.read_csv('dataset_final.csv', delimiter=',',skiprows=counter[k1], nrows=counter_list[k1+1]-counter[k1],names=colnames)
    datasetlist[k1+1]=datasetlist[k1].sort_values(by=["Local_X"], ascending=True)
    datasetlist[k1+1].to_csv("data_"+str(k1+2)+".csv", index=False)
print(counter)
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        