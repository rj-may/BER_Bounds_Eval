import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import math
import pickle
import time


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
from modules.multi_bounds_v3 import bounds_class
from modules.Bhatt_knn_func import knn_num_calc
from modules.data_gen import data_gen


sample_sizes = np.logspace(2, 3.3011, 9 , endpoint = True, dtype = int)


dimension =  8

MC_num = 400

bound_obj_lst = []

func0 = np.random.uniform
func1 = np.random.normal

params0 = {'low': .5, 'high':3}
params1= {"loc":0, "scale" : 1}

generator = data_gen(func0, func1,  params0, params1, dimension)
bound_types =  ["dp", "Bhattacharyya", "Bhatt_knn",  "influence", "enDive"]


for i in sample_sizes:

    start = time.time()
    sample_size =i 
    
    k = knn_num_calc(i, dimension)
    
    if  i < 750:
        threads =2
    else:
        threads = 4

    bounds = bounds_class(generator, sample_size = sample_size, threads =threads, bound_types= bound_types,   MC_num = MC_num, k_nn  =k )
    
    bound_obj_lst.append(bounds)
    
    
    
    
            
    end = time.time()
    
    
    print("done with ", i, " in ",  end -start )

file_path = 'sim_data/uniform_normal8.pkl' # DONT FORGET TO CHANGE ME IF YOU COPY AND PASTE
objects_to_save = bound_obj_lst

with open(file_path, 'wb') as file:
        # Use pickle.dump to serialize and write the list of objects to the file
        pickle.dump(objects_to_save, file)
print(f'Objects saved to {file_path}')
    
