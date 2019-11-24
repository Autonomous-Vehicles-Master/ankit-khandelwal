import Target_and_Surr_Veh_Selection
import pandas as pd
import Data_Processing_and_Filtering
import numpy as np
import sys

def number_of_instances():
    counter=[0 for i in range(Data_Processing_and_Filtering.number_of_vehicle()+1)]
    ds = pd.read_csv('LSTM_Normalized.csv', delimiter=',')
    
    for num in range(0,ds.shape[0],1):
            counter[(ds['Vehicle_ID'][num])]=counter[(ds['Vehicle_ID'][num])]+1
    no_of_instances = min(counter)-4
    return(no_of_instances)
#print(min_no_of_instances)

#Data_Processing_and_Filtering.lstm_data_processing(11) 

def Feature_Matrix_creation(v_id):
    vehicle=Target_and_Surr_Veh_Selection.veh_selection(v_id)
    min_no_of_instances=number_of_instances()
    df_tar=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[0])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_tar['Vehicle_ID'][10]!=vehicle[0]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    x_targ=df_tar['x_hat_n']
    #print(x_targ)
    y_targ=df_tar['Local_Y_n']
    #print(y_targ)
    v_x_targ=df_tar['x_vel_hat_n']
    #print(y_targ)
    v_y_targ=df_tar['y_vel_hat_n']
    #print(y_targ)
    v_type_targ=df_tar['v_Type_n']
    #print(y_targ)
    
    
    df_fl=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[1])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_fl['Vehicle_ID'][10]!=vehicle[1]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_fl=df_fl['x_vel_hat_n']
    #print(v_x_fl)
    delta_v_y_fl=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_fl['y_vel_hat_n'])
    #print(delta_v_y_fl)
    delta_x_fl=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_fl['x_hat_n'])
    #print(delta_x_fl)
    delta_y_fl=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_fl['Local_Y_n'])
    #print(delta_y_fl)
    v_type_fl=df_fl['v_Type_n']
    #print(v_type_fl)
    

    df_ff=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[2])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_ff['Vehicle_ID'][10]!=vehicle[2]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_ff=df_ff['x_vel_hat_n']
    #print(v_x_ff)
    delta_v_y_ff=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_ff['y_vel_hat_n'])
    #print(delta_v_y_ff)
    delta_x_ff=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_ff['x_hat_n'])
    #print(delta_x_ff)
    delta_y_ff=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_ff['Local_Y_n'])
    #print(delta_y_ff)
    v_type_ff=df_ff['v_Type_n']
    #print(v_type_ff)


    df_fr=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[3])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_fr['Vehicle_ID'][10]!=vehicle[3]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_fr=df_fr['x_vel_hat_n']
    #print(v_x_fr)
    delta_v_y_fr=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_fr['y_vel_hat_n'])
    #print(delta_v_y_fr)
    delta_x_fr=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_fr['x_hat_n'])
    #print(delta_x_fr)
    delta_y_fr=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_fr['Local_Y_n'])
    #print(delta_y_fr)
    v_type_fr=df_fr['v_Type_n']
    #print(v_type_fr)

 
    df_l=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[4])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_l['Vehicle_ID'][10]!=vehicle[4]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_l=df_l['x_vel_hat_n']
    #print(v_x_l)
    delta_v_y_l=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_l['y_vel_hat_n'])
    #print(delta_v_y_l)
    delta_x_l=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_l['x_hat_n'])
    #print(delta_x_l)
    delta_y_l=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_l['Local_Y_n'])
    #print(delta_y_l)
    v_type_l=df_l['v_Type_n']
    #print(v_type_l)


    df_f=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[5])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_f['Vehicle_ID'][10]!=vehicle[5]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_f=df_f['x_vel_hat_n']
    #print(v_x_f)
    delta_v_y_f=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_f['y_vel_hat_n'])
    #print(delta_v_y_f)
    delta_x_f=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_f['x_hat_n'])
    #print(delta_x_f)
    delta_y_f=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_f['Local_Y_n'])
    #print(delta_y_f)
    v_type_f=df_f['v_Type_n']
    #print(v_type_f)


    df_r=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[6])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_r['Vehicle_ID'][10]!=vehicle[6]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_r=df_r['x_vel_hat_n']
    #print(v_x_r)
    delta_v_y_r=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_r['y_vel_hat_n'])
    #print(delta_v_y_r)
    delta_x_r=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_r['x_hat_n'])
    #print(delta_x_r)
    delta_y_r=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_r['Local_Y_n'])
    #print(delta_y_r)
    v_type_r=df_r['v_Type_n']
    #print(v_type_r)


    df_bl=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[7])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_bl['Vehicle_ID'][10]!=vehicle[7]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_bl=df_bl['x_vel_hat_n']
    #print(v_x_bl)
    delta_v_y_bl=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_bl['y_vel_hat_n'])
    #print(delta_v_y_bl)
    delta_x_bl=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_bl['x_hat_n'])
    #print(delta_x_bl)
    delta_y_bl=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_bl['Local_Y_n'])
    #print(delta_y_bl)
    v_type_bl=df_bl['v_Type_n']
    #print(v_type_bl)


    df_b=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[8])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_b['Vehicle_ID'][10]!=vehicle[8]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_b=df_b['x_vel_hat_n']
    #print(v_x_b)
    delta_v_y_b=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_b['y_vel_hat_n'])
    #print(delta_v_y_b)
    delta_x_b=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_b['x_hat_n'])
    #print(delta_x_b)
    delta_y_b=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_b['Local_Y_n'])
    #print(delta_y_b)
    v_type_b=df_b['v_Type_n']
    #print(v_type_b)


    df_br=pd.read_csv('Individual_datasets_normalized/data_'+str(vehicle[9])+'.csv', delimiter=',', nrows=min_no_of_instances)
    if(df_br['Vehicle_ID'][10]!=vehicle[9]):
        raise NameError('Data is not sorted properly')
        sys.exit(1)
    v_x_br=df_br['x_vel_hat_n']
    #print(v_x_br)
    delta_v_y_br=Data_Processing_and_Filtering.normalization_of_data(df_tar['y_vel_hat_n']-df_br['y_vel_hat_n'])
    #print(delta_v_y_br)
    delta_x_br=Data_Processing_and_Filtering.normalization_of_data(df_tar['x_hat_n']-df_br['x_hat_n'])
    #print(delta_x_br)
    delta_y_br=Data_Processing_and_Filtering.normalization_of_data(df_tar['Local_Y_n']-df_br['Local_Y_n'])
    #print(delta_y_br)
    v_type_br=df_br['v_Type_n']
    #print(v_type_br)
    

    Feature_Matrix=np.column_stack((y_targ,v_x_targ,v_type_targ,
                    v_x_fl,delta_v_y_fl,delta_x_fl,delta_y_fl,v_type_fl,
                    v_x_ff,delta_v_y_ff,delta_x_ff,delta_y_ff,v_type_ff,
                    v_x_fr,delta_v_y_fr,delta_x_fr,delta_y_fr,v_type_fr,
                    v_x_l,delta_v_y_l,delta_x_l,delta_y_l,v_type_l,
                    v_x_f,delta_v_y_f,delta_x_f,delta_y_f,v_type_f,
                    v_x_r,delta_v_y_r,delta_x_r,delta_y_r,v_type_r,
                    v_x_bl,delta_v_y_bl,delta_x_bl,delta_y_bl,v_type_bl,
                    v_x_b,delta_v_y_b,delta_x_b,delta_y_b,v_type_b,
                    v_x_br,delta_v_y_br,delta_x_br,delta_y_br,v_type_br,
                    ))
    #print(Feature_Matrix[:2,:])
    output_Matrix=np.column_stack((x_targ,v_y_targ))
    return(Feature_Matrix,output_Matrix) #dimention 1869x48
  
