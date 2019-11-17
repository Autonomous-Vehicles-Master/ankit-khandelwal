# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:08:52 2019

@author: Ankit
"""


import pandas as pd
import glob
import os
from scipy.signal import savgol_filter

def lstm_data_processing(win_size_sg_filter):
    dataset_final1 = pd.read_csv('dataset_in_git.csv', delimiter=',')
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
    
    counter_list=[0 for i in range(40)]
    counter_list_vel=[0 for i in range(40)]
    counter=counter_list
    counter1=counter_list
    print(counter_list) 
    print(dataset_final1['Frame_ID'][0])
    
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
    for j in range(0,141281,1):
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
    
    ds1=dst.sort_values(by=["Local_Y"], ascending=True)
    
    if(os.path.isdir('Individual_datasets')==False):
        os.mkdir('Individual_datasets')
    
    ds1.to_csv("Individual_datasets/data_1.csv", index=False)
    
    for k1 in range(39):
        counter[k1+1]=counter[k1+1]+counter[k1]
        datasetlist[k1]=0
        datasetlist[k1]= pd.read_csv('dataset_final.csv', delimiter=',',skiprows=counter[k1], nrows=counter_list[k1+1]-counter[k1],names=colnames, na_values=" ")
        datasetlist[k1+1]=datasetlist[k1].sort_values(by=["Local_Y"], ascending=True)
        datasetlist[k1+1].to_csv("Individual_datasets/data_"+str(k1+2)+".csv", index=False)
    print(counter)
        
    """
    Combine all csv files
    """
    path = 'Individual_datasets'                     # use your path
    all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv("dataset_LSTM.csv", index=False)
    # doesn't create a list, nor does it append to one  
    
    """
    Calculate velocities and update dataset
    """      
    dataset_LSTM1 = pd.read_csv('dataset_LSTM.csv', delimiter=',')
    dataset_LSTM = dataset_LSTM1.sort_values(by=["Vehicle_ID"], ascending=True)
    m=0
    x_vel=[0 for i in range(141281)]
    y_vel=[0 for i in range(141281)]
    Indexer=[0 for i in range(141281)]
    v_Type=[0 for i in range(141281)]
    for j in range(0,141281,1):
        counter_list_vel[(dataset_final['Vehicle_ID'][j]-1)]=counter_list_vel[(dataset_final['Vehicle_ID'][j]-1)]+1
    print(counter_list_vel)
    for k in range(40):
        for l in range(0,counter_list_vel[k],1):
            if(l==0):
                x_vel[m]=0
                y_vel[m]=0
                Indexer[m]=0
            else:
                x_vel[m]=((dataset_LSTM['Local_X'][l]-dataset_LSTM['Local_X'][l-1]))/(1E-3)
                y_vel[m]=((dataset_LSTM['Local_Y'][l]-dataset_LSTM['Local_Y'][l-1]))/(1E-3)
                Indexer[m]=l
            v_Type[m]=dataset_LSTM['v_Class'][l]-2
            m=m+1
    dataset_LSTM['x_Vel']=x_vel
    dataset_LSTM['y_Vel']=y_vel
    dataset_LSTM['Indexer']=Indexer
    dataset_LSTM['v_Type']=v_Type
    dataset_LSTM.to_csv("dataset_LSTM.csv", index=False)
    
    colnames.append('x_Vel')
    colnames.append('y_Vel')
    colnames.append('Indexer')
    
    
    """
    Drop not usefuol data from dataset
    """
    dataset_useful = dataset_LSTM.drop(['Frame_ID',
                                        'Total_Frames',
                                        'v_Width',
                                        'v_Vel',
                                        'v_Acc',
                                        'v_Class',
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
                                        'Location'], axis=1)
    
    dataset_useful.to_csv("dataset_useful.csv", index=False)
    
    """
    Sort individual vehicle dataset again
    """
    colnames_new=['Vehicle_ID', 	
                  'Global_Time',	
                  'Local_X',	
                  'Local_Y',	
                  'Global_X', 	
                  'Global_Y',
                  'v_length',
                  'Lane_ID',	
                  'x_Vel',	
                  'y_Vel',	
                  'Indexer',	
                  'v_Type']
    dst= pd.read_csv('dataset_useful.csv', delimiter=',',nrows=counter_list[0])
    
    ds1=dst.sort_values(by=["Local_Y"], ascending=True)
    
    if(os.path.isdir('Individual_datasets_updated')==False):
        os.mkdir('Individual_datasets_updated')
    
    ds1.to_csv("Individual_datasets_updated/data_1.csv", index=False)
    
    for k1 in range(39):
        datasetlist[k1]=0
        datasetlist[k1]= pd.read_csv('dataset_useful.csv', delimiter=',',skiprows=counter[k1], nrows=counter_list[k1+1]-counter[k1],names=colnames_new, na_values=" ")
        datasetlist[k1+1]=datasetlist[k1].sort_values(by=["Local_Y"], ascending=True)   
        datasetlist[k1+1].to_csv("Individual_datasets_updated/data_"+str(k1+2)+".csv", index=False)
    print(counter1)
    
    """
    Filter noise from velocities and positions
    """ 
    if(os.path.isdir('Individual_datasets_filtered')==False):     
        os.mkdir('Individual_datasets_filtered')
    
    for k1 in range(40):
        #print(datasetlist[k1])
        x_vel_hat=[0 for i in range(5500)]
        y_vel_hat=[0 for i in range(5500)]
        x_hat=[0 for i in range(5500)]
        datasetlist[k1]=0
        datasetlist[k1]= pd.read_csv('Individual_datasets_updated/data_'+str(k1+1)+'.csv', delimiter=',')
        datasetlist[k1] = datasetlist[k1].sort_values(by=["Indexer"], ascending=True)
        x_hat = savgol_filter(datasetlist[k1]['Local_X'], win_size_sg_filter, 3)
        datasetlist[k1]['x_hat']=x_hat     
        
        x_vel_hat= savgol_filter(datasetlist[k1]['x_Vel'], win_size_sg_filter, 3)
        datasetlist[k1]['x_Vel_hat']=x_vel_hat       
        
        y_vel_hat= savgol_filter(datasetlist[k1]['y_Vel'], win_size_sg_filter, 3) 
        datasetlist[k1]['y_Vel_hat']=y_vel_hat       
    
        datasetlist[k1].to_csv("Individual_datasets_filtered/data_"+str(k1+1)+".csv", index=False)
    print(counter1)
                  
    """
    Combine all csv files
    """
    path = 'Individual_datasets_filtered'                     # use your path
    all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv("LSTM_dts.csv", index=False)
    # doesn't create a list, nor does it append to one         

            
lstm_data_processing(11)        
        
        
        
        
        
        
        
        