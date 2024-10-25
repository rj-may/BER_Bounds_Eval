import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import math
import pickle
import time


### set parent directory 
import os
import sys

import matlab.engine


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
from modules.data_gen import data_gen


# sample_sizes = np.logspace(2, 3.3011, 9 , endpoint = True, dtype = int)

start = math.log10(54)
end = math.log10(5000)

sample_sizes = np.logspace(start, end+.00001, 11, dtype=int)

print(sample_sizes)

def main(dim =3):

    
    dim_str= str(dim)
    dimension= int(dim)
    print("Computing beta beta with dimension" + dim_str)


    MC_num = 400

    bound_obj_lst = []

    func0 = np.random.beta
    func1 = np.random.beta

    params0= {'a':2, 'b':5}
    params1 = {'a': 5, 'b':2}


    generator = data_gen(func0, func1,  params0, params1, dimension, boundary=.5)

    eng = matlab.engine.start_matlab()
    eng.cd(r'modules', nargout=0)


    for i in sample_sizes:

        start = time.time()
        sample_size =i 
        
        k = knn_num_calc(i, dimension)
        
        if  i < 250:
            threads = 5
        elif i < 500:
            threads = 10 # was 8
        elif i < 1000:
            threads = 16
        else:
            threads = 20
        bounds = bounds_class(generator, eng,  sample_size = sample_size, threads =threads,  MC_num = MC_num, k_nn  =k )
        
        bound_obj_lst.append(bounds)
        
        
        
        
                
        end = time.time()
        
        
        print("done with ", i, " in ",  end -start )

    file_path = 'sim_data/beta_beta' + dim_str + '.pkl'

    objects_to_save = [bound_obj_lst, sample_sizes]

    with open(file_path, 'wb') as file:
            # Use pickle.dump to serialize and write the list of objects to the file
            pickle.dump(objects_to_save, file)
    print(f'Objects saved to {file_path}')

    eng.quit()


for j in range(1,len(sys.argv) ):
    #  print(sys.argv[j])
     main(sys.argv[j])
