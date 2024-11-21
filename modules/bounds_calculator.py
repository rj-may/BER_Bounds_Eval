from modules.dp_func import get_bounds_dp
from modules.Bhattacharyya_func import get_Bhattacharyya_bounds
from modules.Bhattacharyya_func import get_Maha_upper
from modules.tight_knn_func import __calc_tight_bounds_via_knn_density
from modules.Bhatt_knn_func import __calc_bha_knn_bounds
from modules.knn_density import get_knn_densities

import math

def bounds_calcultor(data0, data1, k_nn=0, alpha_tight=50, kernel='uniform', MATLAB= False):
    """
    Calculate all bounds for a single simulation.

    Args:
        data0: Samples from class 0.
        data1: Samples from class 1.
        k_nn: Number of neighbors for k-NN-based methods.
        alpha_tight: Alpha parameter for tight bounds calculation.
        kernel: Kernel type for methods like enDive.
        handle_errors_dict: Dictionary of error handling settings for bounds methods.

    Returns:
        A dictionary with lower and upper bounds for each calculated bound type.
    """

    results = {}
    
        

    # Calculate DP bounds
    dp_l, dp_u = get_bounds_dp(data0, data1, handle_errors="worst")
    results["dp_lower"] = dp_l
    results["dp_upper"] = dp_u

    # Calculate Bhattacharyya bounds
    bha_l, bha_u = get_Bhattacharyya_bounds(data0, data1, handle_errors="worst")
    results["Bhattacharyya_lower"] = bha_l
    results["Bhattacharyya_upper"] = bha_u

    # Calculate tight bounds
    p0, p1 = get_knn_densities(data0, data1, k_nn)

    prior0 = len(data0) / (len(data0) + len(data1))
    prior1 = len(data1) / (len(data0) + len(data1))

    tight_l, tight_u = __calc_tight_bounds_via_knn_density(p0, p1, prior0, prior1, alpha=alpha_tight)
    results["tight_lower"] = tight_l
    results["tight_upper"] = tight_u

    a, b = __calc_bha_knn_bounds(p0, p1, prior0, prior1,  handle_errors= "worst")
    results["Bha_knn_lower"] = a
    results["Bha_knn_upper"] = b


    # # Calculate Mahalanobis bounds
    # maha_u = get_Maha_upper(data0, data1)
    # results["Mahalanobis_upper"] = maha_u
    # Calculate enDive bounds

    if MATLAB:
        import matlab.engine

        eng = matlab.engine.start_matlab()
        eng.cd(r'modules', nargout=0)
        Dp = eng.EnDive(data0, data1, 'type', "DP", 'quiet', 'kernel', kernel, 'est', 2, nargout=1)
        if Dp > 1:
            Dp = 1
        if Dp > 0:
            enDive_u = 0.5 - 0.5 * Dp
            enDive_l = 0.5 - 0.5 * (Dp ** 0.5)
        else:
            enDive_l, enDive_u = 0.5, 0.5
        results["enDive_lower"] = enDive_l
        results["enDive_upper"] = enDive_u

        estim = eng.hellingerDivergence(data0, data1,[], [],  nargout= 1)
        BC = 1 - estim
        up = 1/2 * BC 
        low = 1/2 - 1/2 * math.sqrt(1- BC**2)

        results["influence_lower"] = low
        results["influence_upper"] = up

        # Quit MATLAB engine
        eng.quit()

    return results
