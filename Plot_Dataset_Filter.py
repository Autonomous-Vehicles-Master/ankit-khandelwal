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
"""
Plot one of the vehicle data to visualize
"""
def plot_dataset_filter(vehicle_id):
    index=[t for t in range(1870)]
    print('Graphs for Vehicle '+str(vehicle_id))
    ds1 = pd.read_csv('Individual_datasets_updated/data_'+str(vehicle_id)+'.csv', delimiter=',', nrows=1870)
    ds2 = pd.read_csv('Individual_datasets_filtered/data_'+str(vehicle_id)+'.csv', delimiter=',', nrows=1870)
    ds1 = ds1.sort_values(by=["Local_Y"], ascending=True)
    ds2 = ds2.sort_values(by=["Local_Y"], ascending=True)    
    
    a=ds1['y_Vel']
    b=ds2['y_Vel_hat']
    
    plt.plot(index,a,c='r')
    plt.plot(index,b)
    plt.xlabel('time in ms') 
    plt.ylabel('Longitudenal Velocity') 
    plt.title('Longitudenal Velocity Comparision')
    plt.legend(['Actual', 'Filtered'], loc='upper left')
    plt.show()   
    
plot_dataset_filter(1)