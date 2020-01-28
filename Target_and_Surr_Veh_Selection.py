# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:47:49 2019

@author: Ankit
"""
import pandas as pd
import Data_Processing_and_Filtering

def veh_selection(target):
    target_veh=0
    vehicle_b=0
    vehicle_ff=0
    vehicle_f=0
    vehicle_r=0
    vehicle_l=0
    vehicle_fl=0
    vehicle_bl=0
    vehicle_fr=0
    vehicle_br=0
    
    vehicle_lane=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    global_time=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    
    vehicle_initial_y_position=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    delta_y_from_targ=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    
    vehicle_initial_x_position=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    delta_x_from_targ=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    
    delta_y_from_front=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    delta_y_from_right=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    delta_y_from_left=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle())]
    
    """
    Selection of Target Vehicle
    """
    target_veh=target
    #print('Target Vehicle is: '+str(target_veh))
    
    """
    Read data required to select other surrounding vehicles
    """
    for p in range (Data_Processing_and_Filtering.number_of_vehicle()):
        dt = pd.read_csv("Individual_datasets_filtered/data_"+str(p+1)+".csv", delimiter=',')
        vehicle_lane[p]=dt['Lane_ID'][0]
        global_time[p]=dt['Global_Time'][0]
        vehicle_initial_y_position[p]=dt['Local_Y'][0]
        vehicle_initial_x_position[p]=dt['x_hat'][0]
        
    #print(vehicle_initial_y_position)
    #print(vehicle_initial_x_position)
    
    """
    selection of vehicle b, f, l and r
    """
    for q in range (Data_Processing_and_Filtering.number_of_vehicle()):
        if(vehicle_lane[q]==vehicle_lane[target_veh-1]):
            if(abs(global_time[target_veh]-global_time[q]) < 12000):
                delta_y_from_targ[q]=vehicle_initial_y_position[q]-vehicle_initial_y_position[target_veh-1]
        if(vehicle_lane[q]!=vehicle_lane[target_veh-1]):
            if(abs(global_time[target_veh]-global_time[q]) < 12000):
                delta_x_from_targ[q]=vehicle_initial_x_position[q]-vehicle_initial_x_position[target_veh-1]
            
    #print(delta_y_from_targ)
    #print(delta_x_from_targ)
    
    try:
        vehicle_b = delta_y_from_targ.index(max([n for n in delta_y_from_targ if n<0]))+1
    except ValueError:
        vehicle_b = 0
        
    try:
        vehicle_f = delta_y_from_targ.index(min([n for n in delta_y_from_targ if n>0]))+1
    except ValueError:
        vehicle_f = 0
        
    try:
        vehicle_l = delta_x_from_targ.index(max([n for n in delta_x_from_targ if n<0]))+1
    except ValueError:
        vehicle_l = 0
        
    try:
        vehicle_r = delta_x_from_targ.index(min([n for n in delta_x_from_targ if n>0]))+1
    except ValueError:
        vehicle_r = 0
        
    #print('f='+str(vehicle_f))
    #print('b='+str(vehicle_b))
    #print('r='+str(vehicle_r))
    #print('l='+str(vehicle_l))
    
    """
    selection of vehicle ff
    """
    if(vehicle_f!=0):
        for r in range (Data_Processing_and_Filtering.number_of_vehicle()):
            if(vehicle_lane[r]==vehicle_lane[vehicle_f-1]):
                if(abs(global_time[target_veh]-global_time[r]) < 12000):
                    delta_y_from_front[r]=vehicle_initial_y_position[r]-vehicle_initial_y_position[vehicle_f-1]
    
    #print(delta_y_from_front)
    
    try:        
        vehicle_ff = delta_y_from_front.index(min([n for n in delta_y_from_front if n>0]))+1
    except ValueError:
        vehicle_ff = 0
        
    #print('ff='+str(vehicle_ff))
    
    """
    selection of vehicle fl and bl
    """
    if(vehicle_l!=0):
        for s in range (Data_Processing_and_Filtering.number_of_vehicle()):
            if(vehicle_lane[s]==vehicle_lane[vehicle_l-1]):
                if(abs(global_time[target_veh]-global_time[s]) < 12000):
                    delta_y_from_left[s]=vehicle_initial_y_position[s]-vehicle_initial_y_position[vehicle_l-1]
            
    #print(delta_y_from_left)
    
    try:
        vehicle_bl = delta_y_from_left.index(max([n for n in delta_y_from_left if n<0]))+1
    except ValueError:
        vehicle_bl = 0
        
    try:    
        vehicle_fl = delta_y_from_left.index(min([n for n in delta_y_from_left if n>0]))+1
    except ValueError:
        vehicle_fl = 0
        
    #print('fl='+str(vehicle_fl))
    #print('bl='+str(vehicle_bl))
    
    """
    selection of vehicle fr and br
    """
    if(vehicle_r!=0):
        for t in range (Data_Processing_and_Filtering.number_of_vehicle()):
            if(vehicle_lane[t]==vehicle_lane[vehicle_r-1]):
                if(abs(global_time[target_veh]-global_time[t]) < 12000):
                    delta_y_from_right[t]=vehicle_initial_y_position[t]-vehicle_initial_y_position[vehicle_r-1]
                
    #print(delta_y_from_right)
    
    try:
        vehicle_br = delta_y_from_right.index(max([n for n in delta_y_from_right if n<0]))+1
    except ValueError:
        vehicle_br = 0
        
    try:    
        vehicle_fr = delta_y_from_right.index(min([n for n in delta_y_from_right if n>0]))+1
    except ValueError:
        vehicle_fr = 0
        
    #print('fr='+str(vehicle_fr))
    #print('br='+str(vehicle_br))

    return(target_veh,vehicle_fl,vehicle_ff,vehicle_fr,vehicle_l,vehicle_f,vehicle_r,vehicle_bl,vehicle_b,vehicle_br)
    













