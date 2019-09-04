import numpy as np
import os
import time
number_of_simulations=10
np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)
number_levels=21
rate_number=4 #Number of available events in the simulation, in this version it contains scd, scdif, adif and cell death
final_output_events=float(3.5e11/2)
gamma=2.4
gamma_progenitor=2.0*gamma
p=0.9
p_stem_cell=0.5
sim_time=10
mu=1e-6
fitness_s=0.1 #fitness of the mutant cells
#Creating the matrix that will contains most of the data, it's organized by row (each row is a level)
#The columns are delta,p,q,rscd, rscdif, acdif
#Uncomment this line to start with an already build up system.
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
        rates_matrix_par[number_levels_par-1,0]=2.0*final_output_events
        rates_matrix_par[number_levels_par-2,0]=1.0*final_output_events
        rates_matrix_par[number_levels_par-1,1]=1.0
        rates_matrix_par[number_levels_par-1,2]=1.0
    #calculate gamma for all levels, starting from TDF level down to the stem cell level
    def calculate_delta():
        for i in range(number_levels_par-3,-1,-1):
            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_par
    def calculate_delta_mod():
        for i in range(number_levels_par-3,prog_levels,-1):
            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_par
        for i in range(prog_levels,-1,-1):
            rates_matrix_par[i,0]=rates_matrix_par[i+1,0]/gamma_progenitor
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
    calculate_delta_mod()
    set_p_values()
    set_q_values()
    set_rscd()
    set_rscdif()
    set_radif()
    print((rates_matrix_par[0,0]/c_set[0])*365)
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

def poison_array(array):
    rows,columns=np.shape(array)
    poison_array=np.zeros((rows,columns))
    for i in range(rows):
        for k in range(columns):
            poison_array[i,k]=np.random.poisson(array[i,k])
    return poison_array
#Stabilizing function that creates a new local rates modifying ignoring the necessary rate for the proper level
def stabilize_and_number_cells(c_par,initial_rates_par,number_levels):
    global delta_t
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
#    print(local_rates)
#    print(np.amax(local_rates[np.where(c>0.),:]))
    delta_t=np.array([1/((np.amax(local_rates[np.where(c>0.),:]))*10)])
#    print(delta_t)
    local_rates=np.multiply(c_par.reshape(c_par.shape[0],1),local_rates)#Modifiyng rates according to actual number
#    print(local_rates)
    local_rates=np.multiply(local_rates,delta_t)
#    print(local_rates)
    return local_rates
#Function that creates the number of necessary blocks to allocate the new mutated cells
def create_new_mutations(level_to_perform_par,number_of_mutations,mutations_needed,number_levels_par):
    print(level_to_perform_par%number_levels_par)
    mutations_already=np.floor(level_to_perform_par/number_levels)
    difference_mutations=number_of_mutations-mutations_already
    mutations_to_create=mutations_needed-difference_mutations
    print(mutations_to_create)
    print(t/365)
    for i in range(0,int(mutations_to_create)):
        add_block_rates(number_levels_par)
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
#function that perform the cellular event in the level, with the outcomes of the KMC step, this function updates the global array c
def perform_event_left(event_to_perform_par,level_to_perform_par,number_levels_par,l_mutations,num_cell_par):
    global c
    global initial_rates
    number_of_mutations=np.shape(initial_rates)[0]/number_levels-1 #Calculating the number of mutations present in the hierarchy
    if c[level_to_perform_par]-num_cell_par>0:
        c[level_to_perform_par]=c[level_to_perform_par]-num_cell_par #Decrease by one the number of cells due to the lost of the current cell
    else:
        c[level_to_perform_par]=0
        num_cell_par=abs(c[level_to_perform_par]-num_cell_par)
    #The first leg is the performance of the events in the left wing of the cellular division
    # If the number of mutations is larger that the current number of blocks, then create a new block calling function except in the tdf level that is excluded
    if level_to_perform_par+l_mutations*number_levels>number_of_mutations*number_levels_par+number_levels_par-1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1:
        create_new_mutations(level_to_perform_par,number_of_mutations,l_mutations,number_levels_par)
    else:
        1
    if event_to_perform_par==0:#perform scd in the corresponding level
        c[level_to_perform_par+l_mutations*number_levels_par]=c[level_to_perform_par+l_mutations*number_levels_par]+num_cell_par
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)==number_levels_par-1: #Do nothing in the case of differentiation in the tdf level
        1
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1:#Performing the differentiation in the tdf level
        c[level_to_perform_par+l_mutations*number_levels_par+1]=c[level_to_perform_par+l_mutations*number_levels_par+1]+num_cell_par
    elif event_to_perform_par==2: #Performing ascd in the left wing
        c[level_to_perform_par+l_mutations*number_levels_par+1]=c[level_to_perform_par+l_mutations*number_levels_par+1]+num_cell_par
    else:
        print('Event not found') #Perform warning message in case of undefined event
def perform_event_right(event_to_perform_par,level_to_perform_par,number_levels_par,r_mutations,num_cell_par):
    global c
    global initial_rates
    #If statement that perform the events in the right wing
    number_of_mutations=np.shape(initial_rates)[0]/number_levels-1 #updating the number of mutations after the performance in the left wing
    #Creating new blocks of mutant cells accordingly to the needs
    if level_to_perform_par+r_mutations*number_levels>number_of_mutations*number_levels_par+number_levels_par-1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1:
        create_new_mutations(level_to_perform_par,number_of_mutations,r_mutations,number_levels_par)
    else:
        1
    if event_to_perform_par==0:#perform scd in the right wing
        c[level_to_perform_par+r_mutations*number_levels_par]=c[level_to_perform_par+r_mutations*number_levels_par]+num_cell_par
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)==number_levels_par-1: #perform scd in the tdf level
        1
    elif event_to_perform_par==1 and (level_to_perform_par%number_levels_par)!=number_levels_par-1: #perform scd in other levels
        c[level_to_perform_par+r_mutations*number_levels_par+1]=c[level_to_perform_par+r_mutations*number_levels_par+1]+num_cell_par
    elif event_to_perform_par==2:# Perform acd in the right wing
        c[level_to_perform_par+r_mutations*number_levels_par]=c[level_to_perform_par+r_mutations*number_levels_par]+num_cell_par
    else:
        print('Event not found') #Perform warning message in case of undefined event

def simulation_step(number_levels_par):
    global initial_rates
    global c
    global t
    global delta_t
    initial_state=stabilize_and_number_cells(c,initial_rates,number_levels_par)
    poison=poison_array(initial_state)
    event_number_list=np.where(poison>0.)
#    print(event_number_list)
    for i in range(np.shape(event_number_list)[1]):
        if event_number_list[0][i]<=number_levels:
            mu_zero=0.0
            num_mut_left,num_cells_left=binomial_pdf(mu_zero,poison[event_number_list][i])
#            print(num_mut_left,num_cells_left)
            num_mut_right,num_cells_right=binomial_pdf(mu_zero,poison[event_number_list][i])
#            print(num_mut_right,num_cells_right)
            for k in range(np.size(num_mut_left)):
                perform_event_left(event_number_list[1][i],event_number_list[0][i],number_levels_par,num_mut_left[k],num_cells_left[k])
            for l in range(np.size(num_mut_right)):
                perform_event_right(event_number_list[1][i],event_number_list[0][i],number_levels_par,num_mut_right[l],num_cells_right[l])
        else:
            num_mut_left,num_cells_left=binomial_pdf(mu,poison[event_number_list][i])
    #        print(num_mut_left,num_cells_left)
            num_mut_right,num_cells_right=binomial_pdf(mu,poison[event_number_list][i])
    #        print(num_mut_right,num_cells_right)
            for k in range(np.size(num_mut_left)):
                perform_event_left(event_number_list[1][i],event_number_list[0][i],number_levels_par,num_mut_left[k],num_cells_left[k])
            for l in range(np.size(num_mut_right)):
                perform_event_right(event_number_list[1][i],event_number_list[0][i],number_levels_par,num_mut_right[l],num_cells_right[l])            
    data=np.concatenate((t,c[21:]),axis=0).reshape(1,c.shape[0]-20)
    with open(id_file, 'a') as f:
        np.savetxt(f,data,fmt='%5.10f')#Saving the array to the file
    t=t+delta_t
def binomial_pdf(mu_par,num_cells):
    array=np.zeros((1))
    no_mut_cells=np.random.binomial(num_cells,mu_par)
#    print(no_mut_cells)
    array[0]=num_cells-no_mut_cells
#    print(array)
    k=no_mut_cells
    while k>0:
        array_temp=np.zeros((1))
        array_temp[0]=np.random.binomial(k,mu_par)
#        print(array_temp)
        k_array=k-array_temp[0]
        array=np.append(array,k_array)
        k=array_temp[0]
    cells=np.arange(0,np.size(array))
    return cells, array
coun_par=0
while coun_par<number_of_simulations:
    start_time = time.time() #Measuring the initial time to test the performance of the simulation.
    rates_matrix=np.zeros((number_levels,rate_number+3))
    #Creating the arrays that will storage the dynamic number of cells and the defined number of cells
    c_set=np.zeros((number_levels))
    c_set[0]=10000
    for i in range(1,number_levels):
        c_set[i]=2.15**i*c_set[0]
    fitness_term=np.zeros(number_levels) #array that will store the fitness values for the different levels
    c=np.copy(c_set)
    #c=np.zeros((number_levels))
    c[0]=9975
    prog_levels=7
    id_file='simu'+str(coun_par)+'.txt' #id file to print the output
    delta_t=np.zeros((1))
    try:
        os.remove(id_file) #Check if the file exists, remove it and create a new one
    except OSError:
        pass
    initial_rates,sanity_p_q=construct_rates(rates_matrix,number_levels,gamma,p,p_stem_cell,c_set) #Calling function contstruct rates
    t=np.zeros((1))
    add_block_rates(number_levels)
    c[21]=25
    #print(c)
    counter=0 #counter of the steps in the KMC
    if sanity_p_q!=0:#Creating sanity check otherwise stop simulation
        while t<sim_time: #Iterating many steps of the KMC
            simulation_step(number_levels)
            if np.amax(c)>1e14:
                print('Too much cells')
                break
            counter+=1
    else:
        print('The definitions of gamma and p are inconsistent (p or q > 1) please select new ones carefully')
    #print(initial_rates)
    print('The number of the KMC steps is:', counter)
    print("--- Total time of the simulation in seconds: %s ---" % (time.time() - start_time)) #Print the total time of the simulation
    coun_par=coun_par+1
