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

# mean_list = np.linspace(.25, 3, 23 , endpoint = True)
mean_list = np.linspace(.25, 3, 12 , endpoint = True)


exact_BER = []

for mean in mean_list:
    # Define dx and x range
    dx = 0.001
    x = np.arange(-10, 10 + dx, dx)

    # Define the normal distributions
    f0 = norm.pdf(x, loc= mean * -1, scale =1)

    f1 = norm.pdf(x, loc=0, scale =1)

    # Calculate the minimum of f0 and f1
    min_f0_f1 = np.minimum(f0, f1)
    # Calculate BER as 0.5 * sum(min(f0, f1) * dx)
    BER = 0.5 * np.nansum(min_f0_f1 * dx)

    exact_BER.append(BER)


### import good stuff
from modules.multi_bounds_v3 import bounds_class
from modules.knn_density import knn_num_calc
from modules.data_gen_mv import data_gen_multivariate

# from modules.data_gen import data_gen


def main(dim ):
    dim = int(dim)
    dim_str= str(dim)
    dimension= int(dim)
    print("Computing normal normal mean separation plot with dimension " + dim_str)
    bound_obj_lst = []

    for j in mean_list:
        start = time.time()

        mean1 = np.zeros(dim)
        covariance1 = np.identity(dim)
        mean2 = np.zeros(dim)
        mean2[0] = j
        covariance2= np.identity(dim)
        
        func0 = np.random.multivariate_normal
        func1 = np.random.multivariate_normal


        params0 = {'mean': mean1, 'cov': covariance1}
        params1  = {'mean': mean2, 'cov': covariance2}

        generator = data_gen_multivariate(func0, func1,  params0, params1, boundary= j/2 )
        
        
        k = knn_num_calc(sample_size, dim)


        bounds = bounds_class(generator, sample_size=  sample_size, threads =2,  MC_num = MC_num, k_nn=k)
        
        bound_obj_lst.append(bounds)
        
        end = time.time()
        
        
        print("done with ", j, " in ",  end -start )


    file_path = 'sim_data/mean_sep'+ dim_str+'.pkl' # DONT FORGET TO CHANGE ME IF YOU COPY AND PASTE
    objects_to_save = [bound_obj_lst, mean_list, exact_BER]

    with open(file_path, 'wb') as file:
            # Use pickle.dump to serialize and write the list of objects to the file
            pickle.dump(objects_to_save, file)
    print(f'Objects saved to {file_path}')
        


for j in range(1,len(sys.argv) ):
    #  print(sys.argv[j])
     main(sys.argv[j])