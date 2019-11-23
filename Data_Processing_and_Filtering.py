# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:08:52 2019

@author: Ankit
"""


import pandas as pd
import glob
import os
from scipy.signal import savgol_filter
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

first_veh=1
last_veh=200
no_of_veh_used=(last_veh-first_veh)+1

def normalization_of_data(data_list):
    series = Series(data_list)
    # prepare data for normalization
    values = series.values
    values = values.reshape((len(values), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    # normalize the dataset and print
    normalized = scaler.transform(values)
    return(normalized)

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
    
    counter_list=[0 for i in range(no_of_veh_used)]
    counter_list_vel=[0 for i in range(no_of_veh_used)]
    #print(counter_list) 
    ds1=[]
    ds2=[] 

    dataset_final=dataset_final1.sort_values(by=["Vehicle_ID"], ascending=True)
    dataset_final.to_csv("dataset_final.csv", index=False)
    for j in range(0,dataset_final.shape[0],1):
        counter_list[(dataset_final['Vehicle_ID'][j]-1)]=counter_list[(dataset_final['Vehicle_ID'][j]-1)]+1 
    #print(counter_list)
    min_no_of_instances = min(counter_list)
    print(min_no_of_instances)
    counter=counter_list
    dst= pd.read_csv('dataset_final.csv', delimiter=',',nrows=counter_list[0])
    
    dst1=dst.sort_values(by=["Local_Y"], ascending=True)
    ds1=dst1[:min_no_of_instances] 
    
    if(os.path.isdir('Individual_datasets')==False):
        os.mkdir('Individual_datasets')
    
    ds1.to_csv("Individual_datasets/data_1.csv", index=False)
    
    for k1 in range(no_of_veh_used-1):
        counter[k1+1]=counter[k1+1]+counter[k1]
        ds1=0
        ds2=0
        ds1=pd.read_csv('dataset_final.csv', delimiter=',',skiprows=counter[k1], nrows=counter_list[k1+1]-counter[k1],names=colnames, na_values=" ")
        ds1=ds1.sort_values(by=["Local_Y"], ascending=True)
        ds2=ds1[:min_no_of_instances] 
        ds2.to_csv("Individual_datasets/data_"+str(k1+2)+".csv", index=False)
    #print(counter)
        
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
    Calculate velocities and update dataset. To calculate velocities time is considered as 1 ms, so only difference of x and y direction. 
    """      
    dataset_LSTM1 = pd.read_csv('dataset_LSTM.csv', delimiter=',')
    dataset_LSTM = dataset_LSTM1.sort_values(by=["Vehicle_ID"], ascending=True)
    m=0
    x_vel=[0 for i in range(dataset_LSTM.shape[0])]
    y_vel=[0 for i in range(dataset_LSTM.shape[0])]
    Indexer=[0 for i in range(dataset_LSTM.shape[0])]
    v_Type=[0 for i in range(dataset_LSTM.shape[0])]
    for j in range(0,dataset_LSTM.shape[0],1):
        counter_list_vel[(dataset_LSTM['Vehicle_ID'][j]-1)]=counter_list_vel[(dataset_LSTM['Vehicle_ID'][j]-1)]+1
    #print(counter_list_vel)
    for k in range(no_of_veh_used):
        for l in range(0,counter_list_vel[k],1):
            if(l==0):
                x_vel[m]=dataset_LSTM['Local_X'][l]
                y_vel[m]=0
                Indexer[m]=0
            else:
                x_vel[m]=(dataset_LSTM['Local_X'][l]-dataset_LSTM['Local_X'][l-1])
                y_vel[m]=(dataset_LSTM['Local_Y'][l]-dataset_LSTM['Local_Y'][l-1])
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
    dst= pd.read_csv('dataset_useful.csv', delimiter=',',nrows=min_no_of_instances)
    
    ds1=dst.sort_values(by=["Local_Y"], ascending=True)
    
    if(os.path.isdir('Individual_datasets_updated')==False):
        os.mkdir('Individual_datasets_updated')
    
    ds1.to_csv("Individual_datasets_updated/data_1.csv", index=False)
    counter1=counter_list_vel
    for k1 in range(no_of_veh_used-1):
        counter1[k1+1]=counter1[k1+1]+counter1[k1]
        ds1=0
        ds2=0
        ds1=pd.read_csv('dataset_useful.csv', delimiter=',',skiprows=counter1[k1], nrows=counter1[k1+1]-counter1[k1],names=colnames_new, na_values=" ")
        ds2=ds1.sort_values(by=["Local_Y"], ascending=True)   
        ds2.to_csv("Individual_datasets_updated/data_"+str(k1+2)+".csv", index=False)
    #print(counter1)
    
    """
    Filter noise from velocities and positions
    """ 
    if(os.path.isdir('Individual_datasets_filtered')==False):     
        os.mkdir('Individual_datasets_filtered')
    
    for k1 in range(no_of_veh_used):
        #print(datasetlist[k1])
        x_vel_hat=[0 for i in range(5500)]
        y_vel_hat=[0 for i in range(5500)]
        x_hat=[0 for i in range(5500)]
        ds1=0
        ds1=pd.read_csv('Individual_datasets_updated/data_'+str(k1+1)+'.csv', delimiter=',')
        ds1=ds1.sort_values(by=["Indexer"], ascending=True)
        x_hat = savgol_filter(ds1['Local_X'], win_size_sg_filter, 3)
        ds1['x_hat']=x_hat     
        
        x_vel_hat= savgol_filter(ds1['x_Vel'], win_size_sg_filter, 3)
        ds1['x_Vel_hat']=x_vel_hat       
        
        y_vel_hat= savgol_filter(ds1['y_Vel'], win_size_sg_filter, 3) 
        ds1['y_Vel_hat']=y_vel_hat  
        ds1=ds1.drop(['Global_Time',	
                      'Local_X',		
                      'Global_X', 	
                      'Global_Y',
                      'v_length',	
                      'x_Vel',	
                      'y_Vel'], axis=1)
    
        ds1.to_csv("Individual_datasets_filtered/data_"+str(k1+1)+".csv", index=False)
                  
    """
    Combine all csv files
    """
    path = 'Individual_datasets_filtered'                     # use your path
    all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv("LSTM_dts.csv", index=False)
    
    """
    Create file with Vehicle_ID=0 to normalize data properly
    """
    
    read_data=pd.read_csv('LSTM_dts.csv', delimiter=',')  
    
    x_hat_min=min(read_data['x_hat'])
    local_y_min=min(read_data['Local_Y'])
    x_vel_hat_min=min(read_data['x_Vel_hat'])
    y_vel_hat_min=min(read_data['y_Vel_hat'])
    v_type_min=min(read_data['v_Type'])
    
    dst= pd.read_csv('Individual_datasets_filtered/data_1.csv', delimiter=',')
    dst= dst.to_csv("Individual_datasets_filtered/data_0.csv", index=False)
    ds0= pd.read_csv('Individual_datasets_filtered/data_0.csv', delimiter=',')
    
    Vehicle_ID_0=[0 for i in range(ds0.shape[0])]
    Local_Y_0=[0 for i in range(ds0.shape[0])]
    Lane_ID_0=[0 for i in range(ds0.shape[0])]
    Indexer_0=[0 for i in range(ds0.shape[0])]
    v_Type_0=[0 for i in range(ds0.shape[0])]
    x_hat_0=[0 for i in range(ds0.shape[0])]
    x_Vel_hat_0=[0 for i in range(ds0.shape[0])]
    y_Vel_hat_0=[0 for i in range(ds0.shape[0])]
    
    for d in range(0,ds0.shape[0],1):
        Vehicle_ID_0[d]=0
        Local_Y_0[d]=local_y_min-1
        Lane_ID_0[d]=0
        Indexer_0[d]=d
        v_Type_0[d]=v_type_min-1
        x_hat_0[d]=x_hat_min-1
        x_Vel_hat_0[d]=x_vel_hat_min-1
        y_Vel_hat_0[d]=y_vel_hat_min-1
     
    ds0['Vehicle_ID']=Vehicle_ID_0
    ds0['Local_Y']=Local_Y_0
    ds0['Lane_ID']=Lane_ID_0
    ds0['Indexer']=Indexer_0
    ds0['v_Type']=v_Type_0
    ds0['x_hat']=x_hat_0
    ds0['x_Vel_hat']=x_Vel_hat_0
    ds0['y_Vel_hat']=y_Vel_hat_0
        
    ds0.to_csv("Individual_datasets_filtered/data_0.csv", index=False)
    
    """
    Combine all csv files again to add data_0.csv file
    """
    path='Individual_datasets_filtered'                     # use your path
    all_files=glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    
    df_from_each_file=(pd.read_csv(f) for f in all_files)
    concatenated_df=pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv("LSTM_dt_to_norm.csv", index=False)
    
    """
    Normalize the data between 0 and 1 to feed to LSTM network
    """
    normalized_data=pd.read_csv('LSTM_dt_to_norm.csv', delimiter=',')    
    
    normalized_data['Local_Y_n']=normalization_of_data(normalized_data['Local_Y'])
    normalized_data['v_Type_n']=normalization_of_data(normalized_data['v_Type'])
    normalized_data['x_hat_n']=normalization_of_data(normalized_data['x_hat'])
    normalized_data['x_vel_hat_n']=normalization_of_data(normalized_data['x_Vel_hat'])
    normalized_data['y_vel_hat_n']=normalization_of_data(normalized_data['y_Vel_hat'])
    
    normalized_data=normalized_data.drop(['Local_Y',	
                                          'v_Type',		
                                          'x_hat', 	
                                          'x_Vel_hat',
                                          'y_Vel_hat'], axis=1)
    normalized_data=normalized_data.sort_values(by=["Vehicle_ID"], ascending=True)
    normalized_data.to_csv("LSTM_Normalized.csv", index=False)
    
    """
    Again create all individual csv with normalized data
    """
    dst_norm= pd.read_csv('LSTM_Normalized.csv', delimiter=',')
    colnames_new1=['Vehicle_ID', 	
                  'Lane_ID',		
                  'Indexer',
                  'Local_Y_n',
                  'v_Type_n',
                  'x_hat_n',
                  'x_vel_hat_n',
                  'y_vel_hat_n']
    
    counter2=[0 for i in range(no_of_veh_used+1)]
    
    for j in range(0,dst_norm.shape[0],1):
        counter2[(dst_norm['Vehicle_ID'][j])]=counter2[(dst_norm['Vehicle_ID'][j])]+1
    #print(counter2)
    
    dst= pd.read_csv('LSTM_Normalized.csv', delimiter=',',nrows=counter2[0])
    
    ds0=dst.sort_values(by=["Indexer"], ascending=True)
    
    if(os.path.isdir('Individual_datasets_normalized')==False):
        os.mkdir('Individual_datasets_normalized')
    
    ds0.to_csv("Individual_datasets_normalized/data_0.csv", index=False)
    
    for k1 in range(0,no_of_veh_used,1):
        counter2[k1+1]=counter2[k1+1]+counter2[k1]
        ds1=0
        ds1= pd.read_csv('LSTM_Normalized.csv', delimiter=',',skiprows=counter2[k1], nrows=counter2[k1+1]-counter2[k1],names=colnames_new1, na_values=" ")
        ds1=ds1.sort_values(by=["Indexer"], ascending=True)   
        ds1.to_csv("Individual_datasets_normalized/data_"+str(k1+1)+".csv", index=False)
    #print(counter2)
    
    
#lstm_data_processing(11)        
        
        
        
        
        
        
        