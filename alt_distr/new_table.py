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
from modules.data_gen_gauss_mix import data_gen_gauss_mix




sample_sizes = [100, 2000]

print(sample_sizes)

def main(dim = 3):

    dim = int(dim)
   
    dim_str= str(dim)
    dimension= int(dim)

    ### first generator object
    mean_sep = 2.56
    params0 = {'means': [[-1 * mean_sep], [mean_sep]], 'covariances':  [ [[1]], [[1]]]}
    params1 = {'mean' : np.zeros(dim), 'cov': np.identity(dim) }
    generator1 = data_gen_gauss_mix(params0, params1, boundary = [-1.55, 1.55] )

    ### second
    
    func0 = np.random.uniform
    func1 = np.random.uniform

    params0 = {'low': 0, 'high':1}
    params1= {"low":.8, "high" : 1.8}

    generator2 = data_gen(func0, func1,  params0, params1, dim, boundary=.9)

    
    ### the third
    func0 = np.random.uniform
    func1 = np.random.normal

    params0 = {'low': .5, 'high':3}
    params1= {"loc":0, "scale" : 1}

    generator3 = data_gen(func0, func1,  params0, params1, dim, boundary=.5)

    
    ### fourth
    func0 = np.random.normal
    func1 = np.random.beta
    params0= {"loc":0, "scale" : 1}
    params1 = {'a': 20, 'b':20}

    generator4 = data_gen(func0, func1,  params0, params1, dim, boundary =[0.3219999999942793, 0.6839999999940787] )


    data_gen_list = [generator1, generator2, generator3, generator4]

    MC_num = 400



    eng = matlab.engine.start_matlab()
    eng.cd(r'modules', nargout=0)

    data_list =[]

    for i in sample_sizes:
  
        bound_obj_lst = []

        for g in range(len(data_gen_list)):
            
            generator = data_gen_list[g]
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
            bounds = bounds_class(generator, eng,  sample_size = sample_size, threads =threads,   MC_num = MC_num, k_nn  =k )
            

            ### fix this part
            bound_obj_lst.append(bounds)
                            
            end = time.time()
            
            
            print("done with ", i, "distr: ", g, " in ",  end -start )
        
        
        data_list.append(bound_obj_lst)



    file_path = 'sim_data/table_data'+ dim_str + '.pkl' # DONT FORGET TO CHANGE ME IF YOU COPY AND PASTE
    objects_to_save = {
    "bound_objects": data_list,
    "sample_sizes": sample_sizes,
    "dimension": dim
    }  

    with open(file_path, 'wb') as file:
            # Use pickle.dump to serialize and write the list of objects to the file
            pickle.dump(objects_to_save, file)
    print(f'Objects saved to {file_path}')

    eng.quit()
        


for j in range(1,len(sys.argv) ):
    #  print(sys.argv[j])
     main(sys.argv[j])
