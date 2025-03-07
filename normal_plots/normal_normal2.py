import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import math
import pickle
import time
import matlab.engine


### set parent directory 
import os
import sys

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


### import good stuff
from modules.multi_bounds_parfor import bounds_class
from modules.knn_density import knn_num_calc
# from modules.data_gen import data_gen
from modules.data_gen_mv import data_gen_multivariate
from modules.constants import MC_num, sample_sizes 



print(sample_sizes)





def main(dim = 3):

    dim_str= str(dim)
    dim = int(dim)

    print("Computing multivariate normal with mean separation of 2.56  and dimension of " + dim_str)
    
    bound_obj_lst = []

    mean_sep = 2.56
    func0 = np.random.multivariate_normal
    func1 = np.random.multivariate_normal

    mean1 = np.zeros(dim)
    covariance1 = np.identity(dim)

    mean2 = np.zeros(dim)
    mean2[0] = mean_sep
        
    params0 = {'mean': mean1, 'cov': covariance1}
    params1  = {'mean': mean2, 'cov': covariance1}
        
    generator  = data_gen_multivariate(func0, func1,  params0, params1, boundary = mean_sep/2 )
    
    eng = matlab.engine.start_matlab()
    eng.cd(r'modules', nargout=0)
    
    for i in sample_sizes:

        start = time.time()
        
        k = knn_num_calc(i, dim)
        
        if  i < 250:
            threads = 5
        elif i < 600:
            threads = 10 # was 8

        else:
            threads = 20
        bounds = bounds_class(generator, eng,  sample_size = i, threads =threads,  MC_num = MC_num, k_nn  =k )
        
        bound_obj_lst.append(bounds)
        

                
        end = time.time()
        
        
        print("done with ", i, " in ",  end -start )

    file_path = 'sim_data/normal_normal'+ dim_str +'.pkl'

    objects_to_save = [bound_obj_lst, sample_sizes]

    with open(file_path, 'wb') as file:
            # Use pickle.dump to serialize and write the list of objects to the file
            pickle.dump(objects_to_save, file)
    print(f'Objects saved to {file_path}')

    eng.quit()

for j in range(1,len(sys.argv) ):
    #  print(sys.argv[j])
     main(sys.argv[j])


