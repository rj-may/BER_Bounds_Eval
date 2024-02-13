'''
this function uses the logic of  Arbitrarily Tight Upper and Lower Bounds  on the Bayesian Probability of Error
and a knn_density calculator for the distributions

 '''
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math

def get_tight_bounds_knn(data0, data1, alpha=50, k=10):

    X = np.concatenate([data0, data1])# merge two class data sets to get our X space

    p = len(data0[0]) ## the dimension of the data sets
    n0 = len(data0)
    n1 = len(data1)


    # Fit k-Nearest Neighbors model and get densitities for data set 1
    knn = NearestNeighbors(n_neighbors=k, algorithm = 'brute')
    knn.fit(data0)
    distances, indices = knn.kneighbors(X) # get distance to the  1,2,... kth nearest neighbor across the space x

    density0 = __knn_density_calc(distances, k, p, n0) ## calculate density based off the distances, k, dim, and sample size

    # Fit k-Nearest Neighbors model and get densitities for data set 2

    knn = NearestNeighbors(n_neighbors=k, algorithm= 'brute')
    knn.fit(data1)
    distances, indices = knn.kneighbors(X)    
    
    density1 = __knn_density_calc(distances, k, p, n1)

    lower, upper = __calc_tight_bounds_via_knn_density(density0, density1, alpha)

    return lower, upper

'''
The following formula uses formula for n- dimesional sphere 
$$ p_k(x)  =\frac{k}{n} \frac{1}{ \frac{\pi^{p/2}}{\Gamma(p/2+1)}  \|x-x_k \|^p}.$$
'''

def __knn_density_calc(distances_matrix, k, p, n): # p is the dimension 
    vec = np.zeros(len(distances_matrix))
        
#     for i, dist_lst in enumerate(distances_matrix):
#         dist = dist_lst[k-1]
#         vol=  __calculate_volume(p, dist)
#         vec[i] =  k /  ( n * vol  )
    for i in range(len(vec)):
        dist = distances_matrix[i][k-1]
        vol=  __calculate_volume(p, dist)
        vec[i] =  k /  ( n * vol  ) ### some people use k-1 for variance purposes
    return vec

def __calculate_volume(d, radius):
    return ((np.pi)**(d/2) ) / math.gamma((d/2) + 1) * (radius**d)
    
def __calc_tight_bounds_via_knn_density(density0, density1, alpha):    
    # fx = 0.5 * (density0 + density1)

    px = density1 / (density0 + density1)
    
    n = len(density0) + len(density1) 
    
    glx = np.mean( g_L(px, alpha)    )
    gux = np.mean( g_U(px, alpha, g_L, g_C)  )
    
    return glx, gux


# these functions are only used in __calc_tight_bounds_via_knn_density
def g_L(p, alpha):
    return 1/alpha * np.log(np.cosh(alpha/2) / np.cosh(alpha * (p - 1/2)))
#     return 1/alpha * np.log((1 + np.exp(-alpha)) / (np.exp(-alpha * p) + np.exp(-alpha * (1 - p))))

def g_C(p):
    return ( 1/2 * np.sin( np.pi * p ) )

def g_U(p, alpha,  g_L, g_C):
    return g_L(p, alpha) + (1 - 2 * g_L(0.5, alpha)) * g_C(p)