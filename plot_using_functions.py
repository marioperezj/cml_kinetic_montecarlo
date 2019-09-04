import numpy as np
import matplotlib.pyplot as plt
def readrow(row, cols):
    data = np.fromstring(row, sep=' ')
    data.resize((cols,))
    return data
num_lev=23
num_mut=5
file='bm_space_1_'
with open(file+'.txt', 'rb') as f:
    data = np.array([readrow(row, (num_mut+1)*num_lev+1) for row in f])
dict_leuke=np.empty((num_lev,),dtype=object)
for i in range(num_lev):
    dict_leuke[i]=[k+i for k in range(num_lev+1,(num_mut+1)*num_lev+1,num_lev)]
total_leuke=np.empty((num_lev,),dtype=object)
for i in range(0,num_lev):
    total_leuke[i]=np.sum(data[:,dict_leuke[i]],axis=1)
long=np.shape(total_leuke[0])[0]
ratio_levels=np.zeros((long,num_lev))
for i in range(long):
    for k in range(num_lev):
        ratio_levels[i,k]=(total_leuke[k][i]/(total_leuke[k][i]+data[i,k+1]))*100
cell_type_percentage=np.zeros((long,num_lev))
total_cells=np.zeros((long))
for i in range(0,long):
    sum_temp=0
    for k in range(0,num_lev):
        sum_temp=sum_temp+total_leuke[k][i]+data[i,k+1]
    total_cells[i]=sum_temp
for i in range(0,long):
    for k in range(0,num_lev):
        cell_type_percentage[i,k]=(data[i,k+1]+total_leuke[k][i])/total_cells[i]
print(cell_type_percentage[0,:])
print(cell_type_percentage[long-1,:])
        
def ratio_per_type(lev_initiate,lev_final):
    plt.figure(figsize=(16,8))
#    plt.xlim(0.8,1.6)
#    plt.yscale('log')
#    plt.ylim(0,10)
    plt.grid(True)
    for i in range(lev_initiate,lev_final):
        plt.scatter(data[:,0]/365.0,cell_type_percentage[:,i]*100,marker=',',s=1,label='level '+str(i%num_lev))
    plt.legend(bbox_to_anchor=(1.0,1.0), borderaxespad=0, fontsize=10)
    plt.tick_params(axis='both',labelsize='xx-large')
    plt.xlabel("Time (years)",fontsize=20)
    plt.ylabel("Ratio of BCR-ABL cells with respect to total",fontsize=20)
    plt.savefig(file+'ratio_type.png')
    plt.close()

def ratio_cml_per_level(lev_initiate,lev_final):
    plt.figure(figsize=(16,8))
    plt.ylim(0,1e2)
    plt.grid(True)
    for i in range(lev_initiate,lev_final):
        plt.scatter(data[:,0]/365.0,ratio_levels[:,i],marker=',',s=10,label='level '+str(i%num_lev))
    plt.legend(bbox_to_anchor=(1.0,1.0), borderaxespad=0, fontsize=10)
    plt.tick_params(axis='both',labelsize='xx-large')
    plt.xlabel("Time (years)",fontsize=20)
    plt.ylabel("Ratio of BCR-ABL cells in the same level",fontsize=20)
    plt.savefig(file+'ratio_bcr_abl.png')
    plt.close()
    
def graph_wt_cells(lev_initiate,lev_final):
    plt.figure(figsize=(16,8))
    plt.yscale('log')
    for i in range(lev_initiate,lev_final):
        plt.scatter(data[:,0]/365.0,data[:,i+1],marker=',',s=10,label='level wildtype'+str(i%num_lev))
    plt.legend(bbox_to_anchor=(1.0,1.0), borderaxespad=0, fontsize=10)
    plt.tick_params(axis='both',labelsize='xx-large')
    plt.xlabel("Time (years)",fontsize=20)
    plt.ylabel("Number of cells per level",fontsize=20)
    plt.savefig(file+'graph_wt_cells.png')
    plt.close()
    
def graph_cml_cells(lev_initiate,lev_final,num_mutation):
    plt.figure(figsize=(16,8))
    plt.yscale('log')
#    plt.xlim(2.5,4.0)
#    plt.ylim(1,1e13)
    for i in range(lev_initiate,lev_final):
        plt.scatter(data[:,0]/365.0,data[:,num_mutation*num_lev+i+1],marker=',',s=1.0,label='level bcr-abl'+str(i%num_lev))
    plt.legend(bbox_to_anchor=(1.0,1.0), borderaxespad=0, fontsize=10)
    plt.tick_params(axis='both',labelsize='xx-large')
    plt.xlabel("Time (years)",fontsize=20)
    plt.ylabel("Number of BCR-ABL cells per level",fontsize=20)
    plt.savefig(file+'graph_cml_cells.png')
    plt.close()

graph_cml_cells(0,23,3)
ratio_cml_per_level(0,23)
ratio_per_type(17,23)
