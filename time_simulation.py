import numpy as np
import os
import time
#import argparse
import math as math
import sys
np.set_printoptions(precision=7)
np.set_printoptions(suppress=True)
#Import different parameters for the simulation from the command line
#parser = argparse.ArgumentParser()
#parser.add_argument("-t", help="Time running simulation", type=float)
#parser.add_argument("-p", help="p vaue for all the levels", type=float)
#parser.add_argument("-gamma",help="Gamma values for the simulation",type=float)
#parser.add_argument("-n",help="Number of levels in the simualtion",type=int)
#parser.add_argument("-idfile",help="Name of the file or path to identift the file",type=str,default='mutation_stop_dif_two_steps')
#parser.add_argument("-mu",help="Mutation rate for the cells",type=float)
#args = parser.parse_args()
#Function that creates the matrix with the rates to be considered in the KMC simulation
#The function returns two items, the first one is the matrix with the rates, the second is a flag to assure that p and q are fixed properly
#The flag is zero if p and q are correct, different of zero otherwise

def print_function(ratios_local,cellularity_par):
    delta.append(np.concatenate([time_sim,[leaking_function],[cellularity_par]]).reshape(1,3))
    data_bm.append(np.concatenate((time_sim,bone_marrow),axis=0).reshape(1,bone_marrow.shape[0]+1))
    data_blood.append(np.concatenate((time_sim,blood),axis=0).reshape(1,blood.shape[0]+1))
    data_ratios.append(np.concatenate((time_sim,ratios_local),axis=0).reshape(1,ratios_local.shape[0]+1))

def construct_rates(rates_matrix_par,number_levels_par,gamma_par,p_par,p_stem_cell_par,homeostasis_par):
    global epsilon
    sanity_p_q=1
    #Set number of cells in the stem cell level separately
    def set_p_value_stem_cell():
        rates_matrix_par[0,1]=p_stem_cell_par
#        print(p_stem_cell_par)
    #Set values for delta, p and q for the boundary levels with particular values
    def set_initial_rates():
        rates_matrix_par[number_levels_par-1,0]=final_output_events
        rates_matrix_par[number_levels_par-1,1]=1.0
        rates_matrix_par[number_levels_par-1,2]=1.0
    #calculate gamma for all levels, starting from TDF level down to the stem cell level
    def calculate_delta():
        for i in range(number_levels_par-3,-1,-1):
            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_par
    def calculate_delta_mod():
        for i in range(number_levels_par-2,mitotic_pool,-1):
            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_par
        for i in range(mitotic_pool,-1,-1):
            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_progenitor
#        for i in range(cfc_pool,-1,-1):
#            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_cfc
    #set p value for progenitors levels
    def set_p_values():
        for i in range(1,mitotic_pool+1,1):
            rates_matrix_par[i,1]=p_prog
        for i in range(mitotic_pool+1,number_levels_par-1,1):
            rates_matrix_par[i,1]=p
    #Set q values for progenitors levels using delta and p
    def set_q_values():
        for i in range(1,number_levels_par-1,1):
            rates_matrix_par[i,2]=2.0*(rates_matrix_par[i-1,0]/rates_matrix_par[i,0])/rates_matrix_par[i,1]
    #Set the rate of csd accordingly to the model
    def set_rscd():
        for i in range(0,number_levels_par,1):
            rates_matrix_par[i,3]=0.5*rates_matrix_par[i,0]*(1-rates_matrix_par[i,2])*rates_matrix_par[i,1]
    #Set the rate of csdif accordingly to the model
    def set_rscdif():
        for i in range(0,number_levels_par,1):
            rates_matrix_par[i,4]=0.5*rates_matrix_par[i,0]*rates_matrix_par[i,1]
#        rates_matrix_par[number_levels_par-2,4]=rates_matrix_par[number_levels_par-1,4]/2
    #Set the rate of acdif accordingly to the model
    def set_radif():
        for i in range(0,number_levels_par,1):
            rates_matrix_par[i,5]=rates_matrix_par[i,0]*(1-rates_matrix_par[i,1])
    def set_leaking():
#        rates_matrix_par[number_levels_par-1,7]=0.1
#        rates_matrix_par[18,7]=0.1
#        rates_matrix_par[17,7]=0.1
#        rates_matrix_par[16,7]=0.1
        for i in range(0,number_levels):
#            rates_matrix_par[i,7]=0.1
#            rates_matrix_par[i,7]=2.0*math.exp(-epsilon*((number_levels_par-1)-i))
            rates_matrix_par[number_levels_par-1-i,7]=1/(1+math.exp(epsilon*(i-2.5)))
#        rates_matrix_par[16,7]=0.1
#        rates_matrix_par[19,7]=20.0
#        rates_matrix_par[number_levels_par-2,7]=rates_matrix_par[number_levels_par-2,3]
    #performing all the cahnges
    set_initial_rates()
    set_p_value_stem_cell()
    calculate_delta_mod()
    set_p_values()
    set_q_values()
    set_rscd()
    set_rscdif()
    set_radif()
    set_leaking()
    #performing a sanity check that p and q are not greater than 1, and changing the flag to appropiate value
    if np.amax(rates_matrix_par[:,1:3])>1.0:
        sanity_p_q=0
    else:
        1
    print(rates_matrix_par[0,0]/homeostasis[0]*365)
    reduced_rates=rates_matrix_par[:,3:4+number_levels] #Creating the list of final reduced rates to use in the monte carlo iteration.
    #dividing by the number of cells per level, resulting in rates per cell
    for i in range(0,number_levels):
        reduced_rates[i,:2]=reduced_rates[i,:2]/homeostasis_par[i]
    reduced_rates[:,4]=reduced_rates[:,4]*rates_matrix_par[number_levels_par-1,4]
#    reduced_rates[0:17,3]=0.1*reduced_rates[0:17,1]
#    reduced_rates[17:number_levels,3]=reduced_rates[17:number_levels,1]
    for i in range(0,18):
        reduced_rates[i,3]=0.005*reduced_rates[i,1]
    reduced_rates[18,3]=0.01*reduced_rates[17,1]
    reduced_rates[19,3]=0.02*reduced_rates[18,1]
    reduced_rates[number_levels-1,3]=reduced_rates[number_levels-1,1]
    return reduced_rates, sanity_p_q
#Function that creates the number of division events necessary in each step of the simulation.
def number_of_events(number_levels,array,leaking_flag):
    global delta_t
    global bm_original_cellularity
    global blood
    global bone_marrow
    global beta
    global leaking_function
    global initial_rates
    current_cellularity=np.sum(bone_marrow)
    cells_parameter=(current_cellularity-4.0*bm_original_cellularity)/bm_original_cellularity
    local_rates=np.copy(initial_rates[:,:])
    if array=='bone_marrow':
        if leaking_flag==1:
            if cells_parameter>0:
#                leaking_function=(cells_parameter)**(beta)
                leaking_function=alpha*(cells_parameter)**(beta)
#                leaking_function=0.1
            local_rates[:,0]=np.zeros((np.size(bone_marrow)))
            local_rates[:,1]=np.zeros((np.size(bone_marrow)))
            local_rates[:,2]=np.zeros((np.size(bone_marrow)))
            local_rates[:,3]=np.zeros((np.size(bone_marrow)))
        else:
#            print(leaking_flag)
#            print(local_rates)
#            print(initial_rates[np.where(bone_marrow>0.),:])
            delta_t=np.array([1/((np.amax(initial_rates[np.where(bone_marrow>0.),:2]))*10)])
#            if bone_marrow[37]>bone_marrow[38] and time_sim>365*3:
#                delta_t=delta_t/30
            leaking_function=0
        rates_summary=initial_rates[:,0]-initial_rates[:,1]-initial_rates[:,4]*leaking_function
        for i in range(1,np.size(blood)):
            if i%number_levels!=0:
                rates_summary[i]=rates_summary[i]+initial_rates[i-1,1]
            else:
                1
        local_rates=np.multiply(bone_marrow.reshape(bone_marrow.shape[0],1),local_rates)
        local_rates=np.multiply(local_rates,delta_t)
        local_rates[:,4]=local_rates[:,4]*(leaking_function)
        local_rates[:,3]=np.zeros((np.size(bone_marrow)))
        local_rates[local_rates<0]=0
        poison_array=np.zeros_like(local_rates,dtype=int)
        list_bone=np.where(local_rates>0)
        for i in range(np.shape(list_bone)[1]):
            poison_array[list_bone[0][i],list_bone[1][i]]=np.random.poisson(local_rates[list_bone[0][i],list_bone[1][i]])
    else:
        local_rates=np.multiply(blood.reshape(blood.shape[0],1),local_rates)
        local_rates=np.multiply(local_rates,delta_t)
        local_rates[:,1]=np.zeros((np.size(blood)))
        local_rates[:,4]=np.zeros((np.size(blood)))
        local_rates[:,0]=np.zeros((np.size(blood)))
        local_rates[local_rates<0]=0
        poison_array=np.zeros_like(local_rates)
        list_blood=np.where(local_rates>0)
        for i in range(np.shape(list_blood)[1]):
            poison_array[list_blood[0][i],list_blood[1][i]]=np.random.poisson(local_rates[list_blood[0][i],list_blood[1][i]])
        rates_summary=local_rates[:,0]-local_rates[:,1]-local_rates[:,4]
        for i in range(1,np.size(blood)):
            if i%number_levels!=0:
                rates_summary[i]=rates_summary[i]+local_rates[i-1,1]
            else:
                1
    return poison_array, rates_summary, cells_parameter
#Function that create a new block of cells for mutants in the global arrays blood and bonemarrow and also append the initial rates array
def add_block_rates(number_levels_par,length):
    #Length is the current length of the initial rates array
    global initial_rates
    global sum_rates
    global bone_marrow
    global blood
    new_array=np.zeros((number_levels_par),dtype=int) #create an empty array of cells
    last_block=np.copy(initial_rates[length-number_levels:length,:]) #Exctracting the last block in the array
#    print(sum_rates[:number_levels-2])
    new_mutated_rates=np.add(last_block[:,0],sum_rates[:]*fitness_s) #Creating new mutated rates block
    mutated_diff=np.add(last_block[:,1],-sum_rates[:]*fitness_s)
    last_block[:,0]=new_mutated_rates #Modifying only the scd rates
    last_block[:,1]=mutated_diff #Modifying the scdiff rates
    last_block[:18,3]=last_block[:18,1]*0.005 #Modifying the death rates in blood.
    last_block[19,3]=last_block[19,1]*0.02
    last_block[18,3]=last_block[18,1]*0.01
    last_block[number_levels-1,3]=last_block[number_levels-1,1]
    initial_rates=np.append(initial_rates,last_block,axis=0) #Modifying the initial rates and c global arrays
    bone_marrow=np.append(bone_marrow,new_array)
    blood=np.append(blood,new_array)
    
def perform_event_blood(array_blood):
    global blood
    for i in range(np.size(array_blood[0,:])):
        diff=blood[array_blood[0,i]]-np.sum(array_blood[2:4,i])
        if diff>=0:
            blood[array_blood[0,i]]=diff
        else:
            blood[array_blood[0,i]]=0
#            print('blood')
#            print(array_blood[0,i])
#            print(blood[array_blood[0,i]])
#            print(np.sum(array_blood[2:4,i]))
            array_blood[2,i]=blood[array_blood[0,i]]
            array_blood[3,i]=0
            array_blood[4,i]=blood[array_blood[0,i]]
            array_blood[5,i]=0
        if array_blood[1,i]==0:
            blood[array_blood[0,i]]=blood[array_blood[0,i]]+array_blood[2,i]+array_blood[4,i]
#            if array_blood[0,i]<=number_levels*5 and (array_blood[3,i]>0 or array_blood[5,i]>0):
            if array_blood[3,i]>0 or array_blood[5,i]>0:
                blood[array_blood[0,i]+number_levels]=blood[array_blood[0,i]+number_levels]+array_blood[3,i]+array_blood[5,i]
        elif array_blood[1,i]==1 and array_blood[0,i]%number_levels!=number_levels-1:
            blood[array_blood[0,i]+1]=blood[array_blood[0,i]+1]+array_blood[2,i]+array_blood[4,i]
#            if array_blood[0,i]<=number_levels*5 and (array_blood[3,i]>0 or array_blood[5,i]>0):
            if array_blood[3,i]>0 or array_blood[5,i]>0:
                blood[array_blood[0,i]+1+number_levels]=blood[array_blood[0,i]+1+number_levels]+array_blood[3,i]+array_blood[5,i]
        elif array_blood[1,i]==2:
            blood[array_blood[0,i]]=blood[array_blood[0,i]]+array_blood[2,i]
            if array_blood[3,i]>0:
                blood[array_blood[0,i]+number_levels]=blood[array_blood[0,i]+number_levels]+array_blood[3,0]
            blood[array_blood[0,i]+1]=blood[array_blood[0,i]+1]+array_blood[4,i]
            if array_blood[5,i]>0:
                blood[array_blood[0,i]+1+number_levels]=blood[array_blood[0,i]+1+number_levels]+array_blood[5,i]
        elif array_blood[1,i]==3:
            1
                
def perform_event_bm(array_marrow):
    global bone_marrow
    global stop_simulation
    diff=0
    for i in range(np.size(array_marrow[0,:])):
        diff=bone_marrow[array_marrow[0,i]]-np.sum(array_marrow[2:4,i])
        if diff>=0:
            if array_marrow[0,i]>16 and array_marrow[0,i]<number_levels:
                1
#                print(array_marrow[0,i],bone_marrow[array_marrow[0,i]],np.sum(array_marrow[2:4,i]),diff,array_marrow[1,i])
            bone_marrow[array_marrow[0,i]]=diff
        if diff<0:
#            if array_marrow[0,i]<number_levels:
#                print('neg_cells')
#                print(array_marrow[0,i],bone_marrow[array_marrow[0,i]],np.sum(array_marrow[2:4,i]),diff,array_marrow[1,i])
#                print(array_marrow[1,i])
#            if array_marrow[0,i]==37:
#                stop_simulation=1
#                print('cells_extincted')
            bone_marrow[array_marrow[0,i]]=0
            array_marrow[2,i]=bone_marrow[array_marrow[0,i]]
            array_marrow[3,i]=0
            array_marrow[4,i]=bone_marrow[array_marrow[0,i]]
            array_marrow[5,i]=0
        if array_marrow[1,i]==0:
            bone_marrow[array_marrow[0,i]]=bone_marrow[array_marrow[0,i]]+array_marrow[2,i]+array_marrow[4,i]
            if array_marrow[3,i]>0 or array_marrow[5,i]>0:
                bone_marrow[array_marrow[0,i]+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[3,i]+array_marrow[5,i]
        elif array_marrow[1,i]==1 and array_marrow[0,i]%number_levels!=number_levels-1:
            bone_marrow[array_marrow[0,i]+1]=bone_marrow[array_marrow[0,i]+1]+array_marrow[2,i]+array_marrow[4,i]
            if array_marrow[3,i]>0 or array_marrow[5,i]>0:
                bone_marrow[array_marrow[0,i]+1+number_levels]=bone_marrow[array_marrow[0,i]+1+number_levels]+array_marrow[3,i]+array_marrow[5,i]
        elif array_marrow[1,i]==2 and array_marrow[0,i]%number_levels!=number_levels-1:
            bone_marrow[array_marrow[0,i]]=bone_marrow[array_marrow[0,i]]+array_marrow[2,i]
            if array_marrow[3,i]>0:
                bone_marrow[array_marrow[0,i]+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[3,i]
            bone_marrow[array_marrow[0,i]+1]=bone_marrow[array_marrow[0,i]+1]+array_marrow[4,i]
            if array_marrow[5,i]>0:
                bone_marrow[array_marrow[0,i]+1+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[5,i]
        elif array_marrow[1,i]==1 and array_marrow[0,i]%number_levels==number_levels-1:
            blood[array_marrow[0,i]]=blood[array_marrow[0,i]]+(array_marrow[2,i]+array_marrow[4,i])
#            if array_marrow[0,i]<=number_levels*5 and (array_marrow[3,i]>0 or array_marrow[5,i]>0):
            if array_marrow[3,i]>0 or array_marrow[5,i]>0:
                1
#                blood[array_marrow[0,i]+1+number_levels]=blood[array_marrow[0,i]+1+number_levels]+array_marrow[3,i]+array_marrow[5,i]
        elif array_marrow[1,i]==2 and array_marrow[0,i]%number_levels==number_levels-1:
            bone_marrow[array_marrow[0,i]]=bone_marrow[array_marrow[0,i]]+array_marrow[2,i]
            if array_marrow[3,i]>0:
                bone_marrow[array_marrow[0,i]+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[3,i]
            blood[array_marrow[0,i]+1]=blood[array_marrow[0,i]+1]+array_marrow[4,i]
            if array_marrow[5,i]>0:
                blood[array_marrow[0,i]+1+number_levels]=blood[array_marrow[0,i]+number_levels]+array_marrow[5,i]
        elif array_marrow[1,i]==4:
            blood[array_marrow[0,i]]=blood[array_marrow[0,i]]+np.sum(array_marrow[2:4,i])
        elif array_marrow[1,i]==3:
            1
#        print(bone_marrow[:number_levels])

#Function that checks if its necessary to create a new block and call the appropiate function for it.   
def produce_new_block(array_bm,array_blood):
    #Check if is necessary to create more blocks or not in each iteration, for each compartment check if there is mutations in the last block
    #Use the auxiliar variables to check if there is at least one mutation then it creates the array
    #otherwise performs nothing
    global blood
    length=np.size(blood)
    aux_bm=0
    aux_blood=0 
    for i in range(np.size(array_bm[0,:])):
#        if array_bm[0,i]>number_levels*5:
#            1
#        else:
        if array_bm[0,i]<=length-1 and array_bm[0,i]>=length-number_levels:
            aux_bm=aux_bm+array_bm[3,i]+array_bm[5,i]
        else:
            1
    for i in range(np.size(array_blood[0,:])):
#        if array_blood[0,i]>number_levels*4:
#            1
#        else:
        if array_blood[0,i]<=length-1 and array_blood[0,i]>=length-number_levels and array_blood[1,i]==0:
            aux_blood=aux_blood+array_blood[3,i]+array_blood[5,i]
        else:
            1
        if array_blood[0,i]<length-1 and array_blood[0,i]>=length-number_levels and array_blood[1,i]==1:
            aux_blood=aux_blood+array_blood[3,i]+array_blood[5,i]
        else:
            1
    if aux_blood>0 or aux_bm>0:
        print(time_sim/365.0)
        add_block_rates(number_levels,length)
#Function that creates the detailed number of mutant cells for each type of event
def new_cells_to_produce(cells_array):
    event_list=np.where(cells_array>0.) #Create an array to storage the number of wt and mutant cells per event and level
    full_list=np.zeros((6,np.size(event_list[0])),dtype=int)
    full_list[0:2,:]=np.array(event_list) #Copy the list of event types and levels
    for i in range(np.shape(full_list)[1]):
        if full_list[0,i]<number_levels:
            #No mutant cells arising from the wild type cells, only from bcr-abl cells
            full_list[3,i]=np.random.binomial(cells_array[full_list[0,i],full_list[1,i]],0)
            full_list[2,i]=cells_array[full_list[0,i],full_list[1,i]]-full_list[3,i]
            full_list[5,i]=np.random.binomial(cells_array[full_list[0,i],full_list[1,i]],0)
            full_list[4,i]=cells_array[full_list[0,i],full_list[1,i]]-full_list[5,i]
        else:
            #Creating the number of mutant and non mutant cells with only one mutation per time step
            full_list[3,i]=np.random.binomial(cells_array[full_list[0,i],full_list[1,i]],mu)
            full_list[2,i]=cells_array[full_list[0,i],full_list[1,i]]-full_list[3,i]
            full_list[5,i]=np.random.binomial(cells_array[full_list[0,i],full_list[1,i]],mu)
            full_list[4,i]=cells_array[full_list[0,i],full_list[1,i]]-full_list[5,i]
#    print(full_list)
#    sorted_array=full_list[1,:].argsort()
#    full_list=full_list[:,sorted_array]
    return full_list

def simulation_step():
    bm_cells_generate,rates_summary_bm,cellularity=number_of_events(number_levels,'bone_marrow',0)
    blood_cells_generate,rates_summary_blood,celullarity=number_of_events(number_levels,'blood',0)
    bm_detailed_cells=new_cells_to_produce(bm_cells_generate)
    blood_detailed_cells=new_cells_to_produce(blood_cells_generate)
    produce_new_block(bm_detailed_cells,blood_detailed_cells)
    perform_event_bm(bm_detailed_cells)
    perform_event_blood(blood_detailed_cells)
    bm_cells_generate,rates_summary_bm,cellularity=number_of_events(number_levels,'bone_marrow',1)
    blood_cells_generate,rates_summary_blood,celullarity=number_of_events(number_levels,'blood',1)
    bm_detailed_cells=new_cells_to_produce(bm_cells_generate)
    blood_detailed_cells=new_cells_to_produce(blood_cells_generate)
    produce_new_block(bm_detailed_cells,blood_detailed_cells)
    perform_event_bm(bm_detailed_cells)
    perform_event_blood(blood_detailed_cells)
    if step%100==0:
        print_function(rates_summary_bm,celullarity)    
start_time = time.time() #Measuring the initial time to test the performance of the simulation.
number_levels=21
rate_number=5 #Number of available events in the simulation, in this version it contains scd, scdif, adif and cell death
final_output_events=int((5e10)) #Number of cells to be created by the final terminally differentiated level
gamma=2.1 #Value of gamma for the non mitotic pool
gamma_progenitor=2.9 #Value of gamma for the mitotic pool
p=1.0 #Value of p for the non mitotic pool
p_prog=1.0 #Value of p for the mitotic pool
p_stem_cell=1.0 #P value for the stem cell
sim_time=365*11
leak_function=0
stop_simulation=0
beta=1.0
epsilon=0.8
num_sim=17
alpha=0.001
mu=8e-06
#while beta<1.6:1
dummy=0
mu_string='{:0.1e}'.format(mu)
epsilon_string='{:0.2f}'.format(epsilon)
alpha_string='{:0.2f}'.format(alpha)
beta_string='{:0.2f}'.format(beta)
directory_name='final_'+'alpha_'+str(alpha_string)+'_beta_'+str(beta_string)+'_mu_'+str(mu_string)+'_modified_'+str(epsilon_string)+'/'
path=os.getcwd()+'/'+directory_name
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed")
    sys.exit()
while dummy<num_sim:
    leak_function=0
    fitness_s=0.15 #fitness of the mutant cells
    id_file_bm=directory_name+'bone_marrow_'+str(dummy)+'.txt' #id file to print the output of the bone marrow compartment
    id_file_blood=directory_name+'blood_'+str(dummy)+'.txt' #id file to print the output of the blood compartment
    id_delta=directory_name+'delta_'+str(dummy)+'.txt'
    id_ratios=directory_name+'ratios_'+str(dummy)+'.txt'
    data_bm=[]
    data_blood=[]
    delta=[]
    data_ratios=[]
    time_sim=np.array([0])
    delta_t=np.zeros((1)) 
    try:
        os.remove(id_file_bm) #Check if the file exists, remove it and create a new one
    except OSError:
        pass
    try:
        os.remove(id_file_blood) #Check if the file exists, remove it and create a new one
    except OSError:
        pass
    try:
        os.remove(id_delta) #Check if the file exists, remove it and create a new one
    except OSError:
        pass
    try:
        os.remove(id_ratios) #Check if the file exists, remove it and create a new one
    except OSError:
        pass
    #Creating the matrix that will contains most of the data, it's organized by row (each row is a level)
    #The columns are delta,p,q,rscd, rscdif, acdif
    rates_matrix=np.zeros((number_levels,rate_number+3))
    #Creating the arrays that will storage the dynamic number of cells and the defined number of cells
    homeostasis=np.zeros((number_levels),dtype=int)
    homeostasis[0]=1e4
    for i in range(1,number_levels):
        homeostasis[i]=np.rint(2.0*homeostasis[i-1])
    bm_original_cellularity=np.sum(homeostasis)
#    print(bm_original_cellularity)
    bone_marrow=np.zeros((number_levels),dtype=int)
    for i in range(number_levels):
        bone_marrow[i]=homeostasis[i]
    bone_marrow[0]=homeostasis[0]-1
    blood=np.zeros((number_levels),dtype=int)
    blood[number_levels-1]=homeostasis[number_levels-1]
#    print(bone_marrow)
#    print(blood)
    mitotic_pool=number_levels-5
    initial_rates,sanity_p_q=construct_rates(rates_matrix,number_levels,gamma,p,p_stem_cell,homeostasis) #Calling function contstruct rates
    sum_rates=np.sum(initial_rates[:,:2],axis=1)
    add_block_rates(number_levels,np.size(blood))
    fitness_s=0.03 #fitness of the mutant cells
#    add_block_rates(number_levels,np.size(blood))
#    add_block_rates(number_levels,np.size(blood))
#    add_block_rates(number_levels,np.size(blood))
#    add_block_rates(number_levels,np.size(blood))
    bone_marrow[number_levels]=1
    step=0
    if sanity_p_q!=0:#Creating sanity check otherwise stop simulation
        while time_sim<sim_time:
#        while blood[number_levels-1]<1e3:            
            if bone_marrow[number_levels]==0:
                print('mutant_extincted')
                break
            if np.sum(bone_marrow)==0:
                print('wild_type_sc_extincted')
                break
            if np.amax(bone_marrow)>1e16 or np.amax(blood)>1e16:
                print('too many cells')
                break
            if np.size(blood)>number_levels*30:
                print('too many mutants')
                break
#            if bone_marrow[number_levels]>bone_marrow[0]:
#                print('negative cells')
#                break
            if stop_simulation==1:
                break
            simulation_step()
            time_sim=time_sim+delta_t
            step=step+1
    else:
        print('The definitions of gamma and p are inconsistent (p or q > 1) please select new ones carefully')
#        print(initial_rates[:,:])
    if time_sim/365>2: 
        with open(id_file_bm, 'w') as f:
            for item in data_bm:
                np.savetxt(f,item,fmt='%15.5f')#Saving the array to the file
        with open(id_file_blood, 'w') as f:
            for item in data_blood:
                np.savetxt(f,item,fmt='%15.5f')#Saving the array to the file
        with open(id_delta, 'w') as f:
            for item in delta:
                np.savetxt(f,item,fmt='%15.5f')#Saving the array to the file
        with open(id_ratios, 'w') as f:
            for item in data_ratios:
                np.savetxt(f,item,fmt='%15.5f')#Saving the array to the file
#    if time_sim/365>3:
#        break
    dummy=dummy+1
#    leaking_function=1
#        initial_rates[:,4]=initial_rates[:,4]*leaking_function
    print(initial_rates[:2*number_levels,:])
#    print(leaking_function)
print(bone_marrow[:])
print(blood)
#    print(homeostasis)
#            print(dummy)
print(step)
print("--- Total time of the simulation in minutes: %s ---" % (time.time() - start_time)) #Print the total time of the simulation
#        print(sum_rates)
#        print(homeostasis)
#        print(mu)
#    if time_sim/365>3.0:
#        break
#    beta=beta+0.25
#    epsilon=epsilon+0.1
#    print(epsilon)
     
