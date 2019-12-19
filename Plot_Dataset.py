# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:52:21 2019

@author: Ankit
"""
import pandas as pd
import random
"""
Plot one of the vehicle data to visualize
"""
def plot_dataset(vehicle_id):
    print('Graphs for Vehicle '+str(vehicle_id))
    ds = pd.read_csv('Individual_datasets_filtered/data_'+str(vehicle_id)+'.csv', delimiter=',')
    ds = ds.sort_values(by=["Indexer"], ascending=True)    
       
    df = pd.DataFrame(ds,columns=['x_hat','Indexer'])
    df.plot(x ='Indexer', y='x_hat', kind = 'line') 
         
    df = pd.DataFrame(ds,columns=['x_Vel_hat','Indexer'])
    df.plot(x ='Indexer', y='x_Vel_hat', kind = 'line') 
       
    df = pd.DataFrame(ds,columns=['y_Vel_hat','Indexer'])
    df.plot(x ='Indexer', y='y_Vel_hat', kind = 'line')  

    df = pd.DataFrame(ds,columns=['Lane_ID','Indexer'])
    df.plot(x ='Indexer', y='Lane_ID', kind = 'line')      
    
plot_dataset(random.randrange(1, 40))