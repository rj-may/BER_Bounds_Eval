import numpy as np
import os
import sys
from scipy.stats import norm
#### I have to code the true values and theoreticals. 
import math
import pickle
import time


# Get the current working directory
current_directory = os.getcwd()
# print("Current Directory:", current_directory)

# Move to the parent directory
parent_directory = os.path.dirname(current_directory)
os.chdir(parent_directory)

# Print the updated working directory
updated_directory = os.getcwd()
print("Updated Directory:", updated_directory)
sys.path.append(updated_directory)

#######
#TODO 

MC_num = 400
sample_size= 500

######

dx = 0.001

diff_lst = np.linspace(0, 5, 10000)


exact_BER = []

for diff in diff_lst:
    # Define dx and x range
    dx = 0.001
    x = np.arange(-10, 10 + dx, dx)

    # Define the normal distributions
    f0 = .5 * norm.pdf(x, loc= diff * -1, scale =1) + .5* norm.pdf(x, loc=diff, scale =1)

    f1 = norm.pdf(x, loc=0, scale =1)

    # Calculate the minimum of f0 and f1
    min_f0_f1 = np.minimum(f0, f1)
    # Calculate BER as 0.5 * sum(min(f0, f1) * dx)
    BER = 0.5 * np.nansum(min_f0_f1 * dx)

    exact_BER.append(BER)


x_BER = np.linspace(0.025,.475, 19, endpoint = True) ## this is the desired mean BER points to be calculated at 

mean_lst = [] 

for j in x_BER:
    index = (np.abs(j - np.array(exact_BER) )).argmin()
    mean_lst.append(diff_lst[index])



### import good stuff
from modules.multi_bounds_v3 import bounds_class
from modules.knn_density import knn_num_calc
from modules.data_gen_gauss_mix import data_gen_gauss_mix

# from modules.data_gen import data_gen


def main(dim ):
    dim = int(dim)
    dim_str= str(dim)
    dimension= int(dim)
    print("Computing bimodal normal BER plot with dimension " + dim_str)
    bound_obj_lst = []

    for j in mean_lst:
        start = time.time()

        params0 = {'means': [[-1 * j], [j]], 'covariances':  [ [[1]], [[1]]]}

        params1 = {'mean' : np.zeros(dim), 'cov': np.identity(dim) }

        generator = data_gen_gauss_mix(params0, params1 )
        
        
        k = knn_num_calc(sample_size, dim)


        bounds = bounds_class(generator, sample_size=  sample_size, threads =2,  MC_num = MC_num, k_nn=k)
        
        bound_obj_lst.append(bounds)
        
        end = time.time()
        
        
        print("done with ", j, " in ",  end -start )


    file_path = 'sim_data/bimodal_normal'+ dim_str+'.pkl' # DONT FORGET TO CHANGE ME IF YOU COPY AND PASTE
    objects_to_save = [bound_obj_lst, mean_lst, x_BER]

    with open(file_path, 'wb') as file:
            # Use pickle.dump to serialize and write the list of objects to the file
            pickle.dump(objects_to_save, file)
    print(f'Objects saved to {file_path}')
        


for j in range(1,len(sys.argv) ):
    #  print(sys.argv[j])
     main(sys.argv[j])