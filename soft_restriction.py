import numpy as np
import os
import time
import argparse
import math as math
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
#Import different parameters for the simulation from the command line
parser = argparse.ArgumentParser()
parser.add_argument("-t", help="Time running simulation", type=float)
parser.add_argument("-p", help="p vaue for all the levels", type=float)
parser.add_argument("-gamma",help="Gamma values for the simulation",type=float)
parser.add_argument("-n",help="Number of levels in the simualtion",type=int)
parser.add_argument("-idfile",help="Name of the file or path to identift the file",type=str,default='mutation_stop_dif_two_steps')
parser.add_argument("-mu",help="Mutation rate for the cells",type=float)
args = parser.parse_args()
start_time = time.time() #Measuring the initial time to test the performance of the simulation.
number_levels=22
rate_number=4 #Number of available events in the simulation, in this version it contains scd, scdif, adif and cell death
final_output_events=int((7e10)) #Number of cells to be created by the final terminally differentiated level
gamma=2.0 #Value of gamma for the non mitotic pool
gamma_progenitor=3.0 #Value of gamma for the mitotic pool
p=1.0 #Value of p for the non mitotic pool
p_prog=1.0 #Value of p for the mitotic pool
p_stem_cell=1.0 #P value for the stem cell
sim_time=365*15
mu=1e-10
leak_par=1e-10
beta=0.5
leaking_function=0
rho=0
counter=0 #counter of the steps in the KMC
fitness_s=0.12 #fitness of the mutant cells
files_name='test'
id_file_bm='bm_'+files_name+'.txt' #id file to print the output of the bone marrow compartment
id_file_blood='blood_'+files_name+'.txt' #id file to print the output of the blood compartment
id_delta='delta.txt'
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
#Creating the matrix that will contains most of the data, it's organized by row (each row is a level)
#The columns are delta,p,q,rscd, rscdif, acdif
rates_matrix=np.zeros((number_levels,rate_number+3))
#Creating the arrays that will storage the dynamic number of cells and the defined number of cells
homeostasis=np.zeros((number_levels),dtype=int)
homeostasis[0]=1e4
for i in range(1,number_levels):
    homeostasis[i]=np.rint(2.0*homeostasis[i-1])
bm_original_cellularity=np.sum(homeostasis[:number_levels-1])
bone_marrow=np.zeros((number_levels),dtype=int)
for i in range(number_levels-1):
    bone_marrow[i]=homeostasis[i]
bone_marrow[0]=9999
blood=np.zeros((number_levels),dtype=int)
blood[number_levels-1]=homeostasis[number_levels-1]
mitotic_pool=17
#Function that creates the matrix with the rates to be considered in the KMC simulation
#The function returns two items, the first one is the matrix with the rates, the second is a flag to assure that p and q are fixed properly
#The flag is zero if p and q are correct, different of zero otherwise
def construct_rates(rates_matrix_par,number_levels_par,gamma_par,p_par,p_stem_cell_par,homeostasis_par):
    sanity_p_q=1
    #Set number of cells in the stem cell level separately
    def set_p_value_stem_cell():
        rates_matrix_par[0,1]=p_stem_cell_par
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
    #Set the rate of acdif accordingly to the model
    def set_radif():
        for i in range(0,number_levels_par,1):
            rates_matrix_par[i,5]=rates_matrix_par[i,0]*(1-rates_matrix_par[i,1])
    def set_leaking():
        for i in range(0,number_levels_par-2,1):
            rates_matrix_par[i,6]=rates_matrix_par[i,4]+rates_matrix_par[i,3]
        rates_matrix_par[number_levels_par-2,6]=rates_matrix_par[number_levels_par-2,3]
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
    print((rates_matrix_par[0,0]*365)/homeostasis[0])
#    print(rates_matrix_par)
    #performing a sanity check that p and q are not greater than 1, and changing the flag to appropiate value
    if np.amax(rates_matrix_par[:,1:3])>1.0:
        sanity_p_q=0
    else:
        1
    reduced_rates=rates_matrix_par[:,3:3+number_levels] #Creating the list of final reduced rates to use in the monte carlo iteration.
    #dividing by the number of cells per level, resulting in rates per cell
    for i in range(0,number_levels):
        reduced_rates[i,:]=reduced_rates[i,:]/homeostasis_par[i]
    return reduced_rates, sanity_p_q
#Function that creates the number of division events necessary in each step of the simulation.
def number_of_events(initial_rates_par,number_levels,array,bm_cellu_par,tot_cellu_par):
    global delta_t
    global blood
    global bone_marrow
    global leak_par
    global beta
    global leaking_function
    global rho
    current_cellularity=np.sum(bone_marrow)
    leaking_function=1/(1+math.exp(-leak_par*(current_cellularity-bm_cellu_par)))
    if current_cellularity>tot_cellu_par:
        rho=math.exp(-0.5*beta*((current_cellularity-tot_cellu_par)/tot_cellu_par))
    else:
        rho=1
    if array=='bone_marrow':
        use_array=bone_marrow
    else:
        use_array=blood
        leaking_function=0
        rho=1
    
    delta_t=np.array([1/((np.amax(initial_rates_par[np.where(bone_marrow>0.),:]))*10)]) #Computing delta based on the blood compartment only
    #it suposse the blood compartment should contain the most quickly dividing cells
    if delta_t==0:
        print('bad_delta')
    local_rates=np.multiply(use_array.reshape(use_array.shape[0],1),initial_rates_par[0:np.shape(use_array)[0],:])#Modifiyng rates according to actual number of cells in each compartment and level
    local_rates=np.multiply(local_rates,delta_t) #Multiply also by the amount of time of the simulation step
    local_rates[:,3]=local_rates[:,3]*(leaking_function)
    local_rates[:,:2]=local_rates[:,:2]*(1-leaking_function)
    local_rates[:,:2]=local_rates[:,:2]*rho
    local_rates[local_rates<0.0]=0.0 #Ignoring any case when the ratios goes to negative in the case of rotating mutations vectors
#    print(local_rates)
    poison_array=np.zeros_like(local_rates)
    for i in range(np.shape(local_rates)[0]):
        for k in range(np.shape(local_rates)[1]):
            poison_array[i,k]=np.random.poisson(local_rates[i,k])
#    if array=='bone_marrow':
#        if np.size(bone_marrow)>59:
#            if initial_rates_par[59,0]!=0:
#        #        print(local_rates)
#                if bone_marrow[59]<10:
#                    print(bone_marrow[59])
#                    print(poison_array[59,:])
#    Calculate for each entry of the matrix the number of events happening in each iteration of the simulation
    return poison_array
#Function that create a new block of cells for mutants in the global arrays blood and bonemarrow and also append the initial rates array
def add_block_rates(number_levels_par,length):
    #Length is the current length of the initial rates array
    global initial_rates
    global sum_rates
    global bone_marrow
    global blood
    new_array=np.zeros((number_levels_par),dtype=int) #create an empty array of cells
    last_block=np.copy(initial_rates[length-number_levels:length,:]) #Exctracting the last block in the array
    new_mutated_rates=np.add(last_block[:,0],sum_rates*fitness_s) #Creating new mutated rates block
    mutated_diff=np.add(last_block[:,1],-sum_rates*fitness_s)
    last_block[:,0]=new_mutated_rates #Modifying only the scd rates
    last_block[:,1]=mutated_diff #MOdifying the scdiff rates
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
            array_blood[2,i]=blood[array_blood[0,i]]
            array_blood[3,i]=0
            array_blood[4,i]=blood[array_blood[0,i]]
            array_blood[5,i]=0
        if array_blood[1,i]==0:
            blood[array_blood[0,i]]=blood[array_blood[0,i]]+array_blood[2,i]+array_blood[4,i]
            if array_blood[3,i]>0 or array_blood[5,i]>0:
                blood[array_blood[0,i]+number_levels]=blood[array_blood[0,i]+number_levels]+array_blood[3,i]+array_blood[5,i]
        elif array_blood[1,i]==1 and array_blood[0,i]%number_levels!=number_levels-1:
            blood[array_blood[0,i]+1]=blood[array_blood[0,i]+1]+array_blood[2,i]+array_blood[4,i]
            if array_blood[3,i]>0 or array_blood[5,i]>0:
                blood[array_blood[0,i]+1+number_levels]=blood[array_blood[0,i]+1+number_levels]+array_blood[3,i]+array_blood[5,i]
        elif array_blood[1,i]==2:
            blood[array_blood[0,i]]=blood[array_blood[0,i]]+array_blood[2,i]
            if array_blood[3,i]>0:
                blood[array_blood[0,i]+number_levels]=blood[array_blood[0,i]+number_levels]+array_blood[3,0]
            blood[array_blood[0,i]+1]=blood[array_blood[0,i]+1]+array_blood[4,i]
            if array_blood[5,i]>0:
                blood[array_blood[0,i]+1+number_levels]=blood[array_blood[0,i]+1+number_levels]+array_blood[5,i]
                
def perform_event_bm(array_marrow):
    global bone_marrow
    for i in range(np.size(array_marrow[0,:])):
        diff=bone_marrow[array_marrow[0,i]]-np.sum(array_marrow[2:4,i])
        if diff>=0:
            bone_marrow[array_marrow[0,i]]=diff
        else:
            bone_marrow[array_marrow[0,i]]=0
            array_marrow[2,i]=bone_marrow[array_marrow[0,i]]
            array_marrow[3,i]=0
            array_marrow[4,i]=bone_marrow[array_marrow[0,i]]
            array_marrow[5,i]=0
        if array_marrow[1,i]==0:
            bone_marrow[array_marrow[0,i]]=bone_marrow[array_marrow[0,i]]+array_marrow[2,i]+array_marrow[4,i]
            if array_marrow[3,i]>0 or array_marrow[5,i]>0:
                bone_marrow[array_marrow[0,i]+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[3,i]+array_marrow[5,i]
        elif array_marrow[1,i]==1 and array_marrow[0,i]%number_levels!=number_levels-2:
            bone_marrow[array_marrow[0,i]+1]=bone_marrow[array_marrow[0,i]+1]+array_marrow[2,i]+array_marrow[4,i]
            if array_marrow[3,i]>0 or array_marrow[5,i]>0:
                bone_marrow[array_marrow[0,i]+1+number_levels]=bone_marrow[array_marrow[0,i]+1+number_levels]+array_marrow[3,i]+array_marrow[5,i]
        elif array_marrow[1,i]==2 and array_marrow[0,i]%number_levels!=number_levels-2:
            bone_marrow[array_marrow[0,i]]=bone_marrow[array_marrow[0,i]]+array_marrow[2,i]
            if array_marrow[3,i]>0:
                bone_marrow[array_marrow[0,i]+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[3,i]
            bone_marrow[array_marrow[0,i]+1]=bone_marrow[array_marrow[0,i]+1]+array_marrow[4,i]
            if array_marrow[5,i]>0:
                bone_marrow[array_marrow[0,i]+1+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[5,i]
        elif array_marrow[1,i]==1 and array_marrow[0,i]%number_levels==number_levels-2:
            blood[array_marrow[0,i]+1]=blood[array_marrow[0,i]+1]+array_marrow[2,i]+array_marrow[4,i]
            if array_marrow[3,i]>0 or array_marrow[5,i]>0:
                blood[array_marrow[0,i]+1+number_levels]=blood[array_marrow[0,i]+1+number_levels]+array_marrow[3,i]+array_marrow[5,i]
        elif array_marrow[1,i]==2 and array_marrow[0,i]%number_levels==number_levels-2:
            bone_marrow[array_marrow[0,i]]=bone_marrow[array_marrow[0,i]]+array_marrow[2,i]
            if array_marrow[3,i]>0:
                bone_marrow[array_marrow[0,i]+number_levels]=bone_marrow[array_marrow[0,i]+number_levels]+array_marrow[3,i]
            blood[array_marrow[0,i]+1]=blood[array_marrow[0,i]+1]+array_marrow[4,i]
            if array_marrow[5,i]>0:
                blood[array_marrow[0,i]+1+number_levels]=blood[array_marrow[0,i]+number_levels]+array_marrow[5,i]
        elif array_marrow[1,i]==3:
            blood[array_marrow[0,i]]=blood[array_marrow[0,i]]+np.sum(array_marrow[2:,i])

#Function that checks if its necessary to create a new block and call the appropiate function for it.   
def produce_new_block(array_bm,array_blood):
    #Check if is necessary to create more blocks or not in each iteration, for each compartment check if there is mutations in the last block
    #Use the auxiliar variables to check if there is at least one mutation then it creates the array
    #otherwise performs nothing
    global blood
    global sim_time
    length=np.size(blood)
    aux_bm=0
    aux_blood=0 
    for i in range(np.size(array_bm[0,:])):
        if array_bm[0,i]<=length-1 and array_bm[0,i]>=length-number_levels:
            aux_bm=aux_bm+array_bm[3,i]+array_bm[5,i]
        else:
            1
    for i in range(np.size(array_blood[0,:])):
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
    return full_list
initial_rates,sanity_p_q=construct_rates(rates_matrix,number_levels,gamma,p,p_stem_cell,homeostasis) #Calling function contstruct rates
sum_rates=np.sum(initial_rates[:,:2],axis=1)
add_block_rates(number_levels,np.size(blood))
#add_block_rates(number_levels,np.size(blood))
#add_block_rates(number_levels,np.size(blood))
#add_block_rates(number_levels,np.size(blood))
bone_marrow[number_levels]=1
def simulation_step(initial_rates):
    global bm_original_cellularity
    global leaking_function
    global rho
    mod_original_cellularity=bm_original_cellularity/0.3
    bm_cells_generate=number_of_events(initial_rates,number_levels,'bone_marrow',mod_original_cellularity,bm_original_cellularity)
    delta=np.array([time_sim,delta_t,leaking_function,rho]).reshape(1,4)
    if counter%100==0:
        with open(id_delta,'a') as f:
            np.savetxt(f,delta,fmt='%5.10f')
    blood_cells_generate=number_of_events(initial_rates,number_levels,'blood',mod_original_cellularity,bm_original_cellularity)
    bm_detailed_cells=new_cells_to_produce(bm_cells_generate)
    blood_detailed_cells=new_cells_to_produce(blood_cells_generate)
    produce_new_block(bm_detailed_cells,blood_detailed_cells)
    perform_event_blood(blood_detailed_cells)
    perform_event_bm(bm_detailed_cells)
    data_bm=np.concatenate((time_sim,bone_marrow),axis=0).reshape(1,bone_marrow.shape[0]+1)
    data_blood=np.concatenate((time_sim,blood),axis=0).reshape(1,blood.shape[0]+1)
    if counter%100==0:  
        with open(id_file_bm, 'a') as f:
            np.savetxt(f,data_bm,fmt='%5.10f')#Saving the array to the file
    if counter%100==0:
        with open(id_file_blood, 'a') as f:
            np.savetxt(f,data_blood,fmt='%5.10f')#Saving the array to the file
counter=0
simulation_step(initial_rates)
if sanity_p_q!=0:#Creating sanity check otherwise stop simulation
    while time_sim<sim_time:
#    while blood[number_levels-1]<1e3:            
        if bone_marrow[number_levels]==0:
            print('mutant_extincted')
            break
        if np.sum(bone_marrow)==0:
            print('wild_type_sc_extincted')
            break
        if np.amax(bone_marrow)>1e25 or np.amax(blood)>1e16:
            print('too many cells')
            break
        simulation_step(initial_rates)
        counter=counter+1
        time_sim=time_sim+delta_t
else:
    print('The definitions of gamma and p are inconsistent (p or q > 1) please select new ones carefully')
print(initial_rates[:,:])
print(bone_marrow)
print(blood)
print('The number of the KMC steps is:', counter)
print("--- Total time of the simulation in seconds: %s ---" % (time.time() - start_time)) #Print the total time of the simulation