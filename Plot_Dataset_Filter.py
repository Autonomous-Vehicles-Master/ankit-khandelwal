# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:56:46 2019

@author: Ankit
"""

"""
Created on Sun Nov 10 11:52:21 2019

@author: Ankit
"""
import pandas as pd
from matplotlib import pyplot as plt
import Feature_Matrix_creation
"""
Plot one of the vehicle data to visualize
"""
def plot_dataset_filter(vehicle_id):
    index=[t for t in range(Feature_Matrix_creation.number_of_instances())]
    print('Graphs for Vehicle '+str(vehicle_id))
    ds11 = pd.read_csv('Individual_datasets_updated/data_'+str(vehicle_id)+'.csv', delimiter=',', nrows=Feature_Matrix_creation.number_of_instances())
    ds12 = pd.read_csv('Individual_datasets_filtered/data_'+str(vehicle_id)+'.csv', delimiter=',', nrows=Feature_Matrix_creation.number_of_instances())
    ds11 = ds11.sort_values(by=["Local_Y"], ascending=True)
    ds12 = ds12.sort_values(by=["Local_Y"], ascending=True)    
    
    a=ds11['y_Vel']
    b=ds12['y_Vel_hat']
    
    plt.plot(index,a,c='r')
    plt.plot(index,b)
    plt.xlabel('time in 100ms') 
    plt.ylabel('Longitudenal Velocity') 
    plt.title('Longitudenal Velocity Comparision')
    plt.legend(['Actual', 'Filtered'], loc='upper left')
    plt.show()   
    
plot_dataset_filter(1)