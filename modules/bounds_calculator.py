from modules.dp_func import get_bounds_dp
from modules.Bhattacharyya_func import get_Bhattacharyya_bounds
from modules.Bhattacharyya_func import get_Maha_upper
from modules.tight_knn_func import __calc_tight_bounds_via_knn_density
from modules.Bhatt_knn_func import __calc_bha_knn_bounds

from oct2py import Oct2Py

import time

import math

def bounds_calculator(data0, data1, k_nn=0, alpha_tight=50, kernel='uniform', MATLAB= None, Timer = False,  scikitlearn = False):
    """
    Calculate all bounds for a single simulation.

    Args:
        data0: Samples from class 0.
        data1: Samples from class 1.
        k_nn: Number of neighbors for k-NN-based methods.
        alpha_tight: Alpha parameter for tight bounds calculation.
        kernel: Kernel type for methods like enDive.
        MATLAB: engine that is used to calculate MATLAB functions
        Timer: Decides if we should return a timer also. (WARNING creates a second return)
        scikitlearn : if you have the sckilearn package and want to use that instead use it. 

    Returns:
        A dictionary with lower and upper bounds for each calculated bound type.
    """
        # Calculate tight bounds
    if scikitlearn:
        from modules.knn_density import get_knn_densities

    else:
        from modules.knn_density_scipy import get_knn_densities

    results = {}
    
    if Timer:
        start = time.time()
        timer_results = {}

    # Calculate DP bounds
    dp_l, dp_u = get_bounds_dp(data0, data1, handle_errors="worst")
    results["dp_lower"] = dp_l
    results["dp_upper"] = dp_u

    if Timer:
        timer_results["Dp"] = time.time() - start
        start = time.time()
    

    # Calculate Bhattacharyya bounds
    bha_l, bha_u = get_Bhattacharyya_bounds(data0, data1, handle_errors="worst")
    results["Bhattacharyya_lower"] = bha_l
    results["Bhattacharyya_upper"] = bha_u

    if Timer:
        timer_results["Bhattacharyya"] = time.time() - start
        start = time.time()
    


    ## Calculating the functions with the k-NN densities
    p0, p1 = get_knn_densities(data0, data1, k_nn)

    prior0 = len(data0) / (len(data0) + len(data1))
    prior1 = len(data1) / (len(data0) + len(data1))

    if Timer:
        knn_time = time.time() - start
        start = time.time()


    tight_l, tight_u = __calc_tight_bounds_via_knn_density(p0, p1, prior0, prior1, alpha=alpha_tight)
    results["tight_lower"] = tight_l
    results["tight_upper"] = tight_u

    if Timer:
        # timer_results["tight"]  = time.time() - start + knn_time
        start = time.time()
    

    a, b = __calc_bha_knn_bounds(p0, p1, prior0, prior1,  handle_errors= "worst")
    results["Bha_knn_lower"] = a
    results["Bha_knn_upper"] = b
    
    if Timer:
        timer_results["Bha_knn"]  = time.time() - start + knn_time
        start = time.time()


    # # Calculate Mahalanobis bounds
    # maha_u = get_Maha_upper(data0, data1)
    # results["Mahalanobis_upper"] = maha_u
    # Calculate enDive bounds

    if (MATLAB is not None) or True:
        import matlab.engine

        if isinstance(MATLAB, matlab.engine.MatlabEngine) or True:
            print("here")

            # eng =  MATLAB

            # eng.addpath('modules', nargout=0)

            oc = Oct2Py()
            oc.addpath('modules')  # Equivalent to eng.addpath
            oc.eval("pkg load statistics")
            oc.eval("pkg load optim")
            oc.eval("pkg load geometry")


            if Timer:
                start = time.time()
            
            Dp = oc.EnDive(data0, data1, 'type', "DP", 'quiet', 'kernel', kernel, 'est', 2, nargout=1)  
            # Dp = oc.EnDive(data0, data1,
            #    'quiet',  # bare flag, not key-value
            #    func_args={
            #        'type': "DP",
            #        'kernel': kernel,
            #        'est': 2
            #    },
            #    nargout=1)

            p = prior0
            q = prior1
            up = 4 * p * q * Dp  + (p-q)**2
            if up > 1:
                up = 1
            if up > 0:
                enDive_u = 0.5 - 0.5 * up
                enDive_l = 0.5 - 0.5 * (up ** 0.5)
            else:
                enDive_l, enDive_u = 0.5, 0.5
            results["enDive_lower"] = enDive_l
            results["enDive_upper"] = enDive_u

            if Timer:
                timer_results["enDive"] = time.time() - start
                start = time.time()

            estim = oc.hellingerDivergence(data0, data1,[], [],  nargout= 1)
            BC = 1 - estim

            if  4 * prior0 * prior1 * BC**2  >1:
                print("BC coefficent is estimated bo greater than one for influuence bound. ")
                up= .5
                low = .5
            else: 
                # print(BC, estim)
                up = math.sqrt(prior0 * prior1) * BC 
                low = 1/2 - 1/2 * math.sqrt(1- 4 * prior0 * prior1 * BC**2)

            results["influence_lower"] = low
            results["influence_upper"] = up

            if Timer:
                timer_results["influence"] = time.time() - start
                start = time.time()

        else:
            print("A MATLAB engine was not provided. ")

    if Timer:
        return results, timer_results
    else:
        return results
