#This code was created by Mario Perez at ELTE University, it creates a Kinetic Monte Carlo simulation of a tissue.
#This tissue is based on the following article https://www.nature.com/articles/ncomms14545 when the parameters definitions can be found
#The sumlation outputs a file with the following data in columns
#Time /t delta_time /t Number of cells in each level /t Level to affect /t Event type to affect
import numpy as np
import os
import time
import argparse
np.set_printoptions(precision=7)
np.set_printoptions(suppress=True)
#Import different parameters for the simulation from the command line
parser = argparse.ArgumentParser()
parser.add_argument("-t", help="Time running simulation", type=int)
parser.add_argument("-p", help="p vaue for all the levels", type=float)
parser.add_argument("-gamma",help="Gamma values for the simulation",type=float)
parser.add_argument("-n",help="Number of levels in the simualtion",type=int)
parser.add_argument("-idfile",help="Name of the file or path to identift the file",type=str,default='out_file_kmc')
parser.add_argument("-mu",help="Mutation rate for the cells",type=float)
args = parser.parse_args()
start_time = time.time() #Measuring the initial time to test the performance of the simulation.
number_levels=args.n
rate_number=4 #Number of available events in the simulation, in this version it contains scd, scdif, adif and cell death
gamma=args.gamma
p=args.p
p_stem_cell=0.5
sim_time=args.t
mu=args.mu
counter=0 #counter of the steps in the KMC
fitness_s=0.4 #fitness of the mutant cells
id_file=args.idfile+'.txt' #id file to print the output
try:
    os.remove(id_file) #Check if the file exists, remove it and create a new one
except OSError:
    pass
#Creating the matrix that will contains most of the data, it's organized by row (each row is a level)
#The columns are delta,p,q,rscd, rscdif, acdif
rates_matrix=np.zeros((number_levels,rate_number+3))
#Creating the arrays that will storage the dynamic number of cells and the defined number of cells
c_set=np.array([30,40,50,60,70])
fitness_term=np.zeros(number_levels) #array that will store the fitness values for the different levels
c=np.array([30,0,0,0,0]) #Starting with only the stem cell pool
#c=np.copy(c_set) #Uncomment this line to start with an already build up system.
#Function that creates the matrix with the rates to be considered in the KMC simulation
#The function returns two items, the first one is the matrix with the rates, the second is a flag to assure that p and q are fixed properly
#The flag is zero if p and q are correct, different of zero otherwise
def construct_rates(rates_matrix_par,number_levels_par,gamma_par,p_par,p_stem_cell_par,c_set_par):
    sanity_p_q=1
    #Set number of cells in the stem cell level separately
    def set_p_value_stem_cell():
        rates_matrix_par[0,1]=p_stem_cell_par
    #Set values for delta, p and q for the boundary levels with particular values
    def set_initial_rates():
        rates_matrix_par[number_levels_par-1,0]=2.0
        rates_matrix_par[number_levels_par-2,0]=1.0
        rates_matrix_par[number_levels_par-1,1]=1.0
        rates_matrix_par[number_levels_par-1,2]=1.0
    #calculate gamma for all levels, starting from TDF level down to the stem cell level
    def calculate_delta():
        for i in range(number_levels_par-3,-1,-1):
            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_par
    #set p value for progenitors levels
    def set_p_values():
        for i in range(1,number_levels_par-1,1):
            rates_matrix_par[i,1]=p_par
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
    #performing all the cahnges
    set_initial_rates()
    set_p_value_stem_cell()
    calculate_delta()
    set_p_values()
    set_q_values()
    set_rscd()
    set_rscdif()
    set_radif()
    #performing a sanity check that p and q are not greater than 1, and changing the flag to appropiate value
    if np.amax(rates_matrix_par[:,1:3])>1.0:
        sanity_p_q=0
    else:
        1
    reduced_rates=rates_matrix_par[:,3:3+rate_number] #Creating the list of final reduced rates to use in the monte carlo iteration.
    #dividing by the number of cells per level, resulting in rates per cell
    for i in range(0,number_levels):
        reduced_rates[i,0:rate_number-1]=reduced_rates[i,0:rate_number-1]/c_set_par[i]
    return reduced_rates, sanity_p_q
#function that perform the cellular event in the level, with the outcomes of the KMC step, this function updates the global array c
def perform_event_new(event_to_perform_par,level_to_perform_par,number_levels_par,l_mutations,r_mutations):
    global c
    global initial_rates
    number_of_mutations=np.shape(initial_rates)[0]/number_levels-1 #Calculating the number of mutations present in the hierarchy
    c[level_to_perform_par]=c[level_to_perform_par]-1 #Decrease by one the number of cells due to the lost of the current cell
    #The first leg is the performance of the events in the left wing of the cellular division
    # If the number of mutations is larger that the current number of blocks, then create a new block calling function except in the tdf level that is excluded
    if level_to_perform_par+l_mutations*number_levels>number_of_mutations*number_levels_par+number_levels_par-1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1:
        create_new_mutations(level_to_perform_par,number_of_mutations,l_mutations,number_levels_par)
    else:
        1
    if event_to_perform_par==0:#perform scd in the corresponding level
        c[level_to_perform_par+l_mutations*number_levels_par]=c[level_to_perform_par+l_mutations*number_levels_par]+1
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)==number_levels_par-1: #Do nothing in the case of differentiation in the tdf level
        1
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1:#Performing the differentiation in the tdf level
        c[level_to_perform_par+l_mutations*number_levels_par+1]=c[level_to_perform_par+l_mutations*number_levels_par+1]+1
    elif event_to_perform_par==2: #Performing ascd in the left wing
        c[level_to_perform_par+l_mutations*number_levels_par+1]=c[level_to_perform_par+l_mutations*number_levels_par+1]+1
    else:
        print('Event not found') #Perform warning message in case of undefined event
    #If statement that perform the events in the right wing
    number_of_mutations=np.shape(initial_rates)[0]/number_levels-1 #updating the number of mutations after the performance in the left wing
    #Creating new blocks of mutant cells accordingly to the needs
    if level_to_perform_par+r_mutations*number_levels>number_of_mutations*number_levels_par+number_levels_par-1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1:
        create_new_mutations(level_to_perform_par,number_of_mutations,r_mutations,number_levels_par)
    else:
        1
    if event_to_perform_par==0:#perform scd in the right wing
        c[level_to_perform_par+r_mutations*number_levels_par]=c[level_to_perform_par+r_mutations*number_levels_par]+1
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)==number_levels_par-1: #perform scd in the tdf level
        1
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1: #perform scd in other levels
        c[level_to_perform_par+r_mutations*number_levels_par+1]=c[level_to_perform_par+r_mutations*number_levels_par+1]+1
    elif event_to_perform_par==2:# Perform acd in the right wing
        c[level_to_perform_par+r_mutations*number_levels_par]=c[level_to_perform_par+r_mutations*number_levels_par]+1
    else:
        print('Event not found') #Perform warning message in case of undefined event
#Stabilizing function that creates a new local rates modifying ignoring the necessary rate for the proper level
def stabilize_and_number_cells(c_par,initial_rates_par,number_levels):
    local_rates=np.copy(initial_rates_par) #New local rates
    stem_index=[i for i in range(0,len(c_par),number_levels)] #Index of stem cell levels
    scd_index=[0 for i in range(0,len(c_par),number_levels)]  #Index od scd rates
    scdif_index=[1 for i in range(0,len(c_par),number_levels)] #Index of scdif rates
    number_stem_cells=np.sum(c[stem_index]) #Number of combined stem cells, considering wild type and mutations
    if number_stem_cells>c_set[0]: #Ignoring scd when we have excess of stem cells and keep constant the rate of stem cell
        local_rates[stem_index,scd_index]=scd_index
        local_rates[stem_index,scdif_index]=local_rates[stem_index,scdif_index]*2
    elif number_stem_cells==c_set[0]: #Do nothing if there is the proper number of cells
        1
    elif number_stem_cells<c_set[0]: #Ignore the scdif for lack of stem cells multiplying by two the wild type rate of scd, but not the fitness term.
        local_rates[stem_index,scdif_index]=scd_index
        local_rates[stem_index,scd_index]=local_rates[stem_index,scd_index]*2-fitness_term[stem_index]
    initial_state=np.multiply(c_par.reshape(c_par.shape[0],1),local_rates) #Modifiyng rates according to actual number
    return initial_state
#Function that create a new block of cells for mutants in the global array c
def add_block_rates(number_levels_par):
    global initial_rates
    global c
    global fitness_s
    global fitness_term
    new_c=np.zeros((number_levels_par)) #create an empty array of cells
    length=np.shape(initial_rates)[0] #Calculating the length of the rates array
    last_block=np.copy(initial_rates[length-number_levels:length,:]) #Exctracting the last block in the array
    sum_rates=(np.sum(last_block,axis=1)*fitness_s) #Sum of all the rates in the level
    non_zero_initial=np.where(last_block[:,0]>0)
    non_zero_initial=np.isin(last_block[:,0],non_zero_initial,invert=True) #Ignoring rates with zero probability to not create new arrays
    new_mutated_rates=np.add(last_block[:,0],sum_rates,where=non_zero_initial) #Creating new mutated rates block
    fitness_term=np.append(fitness_term,sum_rates)
    last_block[:,0]=new_mutated_rates #Modifying only the scd rates
    initial_rates=np.append(initial_rates,last_block,axis=0) #Modifying the initial rates and c global arrays
    c=np.append(c,new_c,axis=0)
#Function that creates the number of necessary blocks to allocate the new mutated cells
def create_new_mutations(level_to_perform_par,number_of_mutations,mutations_needed,number_levels_par):
    mutations_already=np.floor(level_to_perform_par/number_levels)
    difference_mutations=number_of_mutations-mutations_already
    mutations_to_create=mutations_needed-difference_mutations
    for i in range(0,int(mutations_to_create)):
        add_block_rates(number_levels_par)
#Function that performs one step of the KMC it receives as arguments c, and initial state the number of levels and time
#The function update the global variable c using perform_event and the time also choose one of the events using the proper algorithm
def kinetic_montecarlo_step(c_par,number_levels_par,t_par):
    #In each event the simulation has two division events, each of one can mutate independently, thus there are two wings, event drwan from a Poisson distribution
    l_mutations=np.random.poisson(mu,1)
    r_mutations=np.random.poisson(mu,1)
    initial_state=stabilize_and_number_cells(c,initial_rates,number_levels) #Creating the initial state with the stabilization method
    rates_indices=np.where(initial_state>0.)#Selecting only non zero rates
    states_list=initial_state[rates_indices]# Getting the indices for the non zero states
    final_rates=np.cumsum(states_list)#create the cumulative sum of the rates
    max_rate=np.amax(final_rates)#getting the maximum of the sum to multiply the random number
    u=np.random.uniform(0,1,1)
    binary_result=np.searchsorted(final_rates,u*max_rate) #Performing binary search of the correct event
    level_to_perform=rates_indices[0][binary_result] #Getting the desired level to affect
    event_to_perform=rates_indices[1][binary_result] #Getting the rigth event to perform
    perform_event_new(event_to_perform,level_to_perform,number_levels_par,l_mutations,r_mutations) #Performing the event
    u_new=np.random.uniform(0,1,1)
    delta_t=np.log(1/u_new)/max_rate #get delta_t
    data=np.concatenate((t_par,delta_t,level_to_perform,event_to_perform,l_mutations,r_mutations,c),axis=0).reshape(1,c.shape[0]+6) #CReating the proper array to print
    if 1==1:
        with open(id_file, 'a') as f:
            np.savetxt(f,data,fmt='%5.10f')#Saving the array to the file
    global t
    t=t_par+delta_t #Update the global varibale t
t=np.zeros((1))
initial_rates,sanity_p_q=construct_rates(rates_matrix,number_levels,gamma,p,p_stem_cell,c_set) #Calling function contstruct rates
print(initial_rates)
if sanity_p_q!=0:#Creating sanity check otherwise stop simulation
    while t<sim_time: #Iterating many steps of the KMC
        kinetic_montecarlo_step(c,number_levels,t)
        counter+=1
else:
    print('The definitions of gamma and p are inconsistent (p or q > 1) please select new ones carefully')
print('The number of the KMC steps is:', counter)
print("--- Total time of the simulation in seconds: %s ---" % (time.time() - start_time)) #Print the total time of the simulation
