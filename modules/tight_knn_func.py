'''
this function uses the logic of  Arbitrarily Tight Upper and Lower Bounds  on the Bayesian Probability of Error
and a knn_density calculator for the distributions

 '''


import numpy as np
import math
try:
    from modules.knn_density import get_knn_densities

except ImportError:
    from modules.knn_density_scipy import get_knn_densities


def get_tight_bounds_knn(data0, data1, alpha=50, k=0):

    ## this is equivalent to the Bhattacharyya distance but we are using the knn densities
    p0, p1 =  get_knn_densities(data0, data1, k)
    
    prior_c0 = len(data0) /  (len(data0) +  len(data1))### probability of each class
    prior_c1 = len(data1) / (len(data0) +  len(data1))

    lower, upper = __calc_tight_bounds_via_knn_density(p0, p1, prior_c0, prior_c1,  alpha)

    return lower, upper


def __calc_tight_bounds_via_knn_density(density0, density1, prior_c0, prior_c1,  alpha): 
    d1 = density1
    d0 = density0 

    fx = prior_c0 * d0 + prior_c1 * d1

    px = d1 / (d0 + d1)

    ### do I need to do this?
    px = np.minimum(px, 1- px )
    
    # n = len(density0) + len(density1) 
    
    glx = np.sum( g_L(px, alpha)  * fx  )
    gux = np.sum( g_U(px, alpha, g_L, g_C) *fx )
    
    return glx, gux


# these functions are only used in __calc_tight_bounds_via_knn_density
def g_L(p, alpha):
    return 1/alpha * np.log(np.cosh(alpha/2) / np.cosh(alpha * (p - 1/2)))
#     return 1/alpha * np.log((1 + np.exp(-alpha)) / (np.exp(-alpha * p) + np.exp(-alpha * (1 - p))))

def g_C(p):
    return ( 1/2 * np.sin( np.pi * p ) )

def g_U(p, alpha,  g_L, g_C):
    return g_L(p, alpha) + (1 - 2 * g_L(0.5, alpha)) * g_C(p)

