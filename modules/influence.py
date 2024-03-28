
"""
This code estimates the hellinger divergence from influence functions. 


Disclaimer:
This Python file is based on code from a library that is not my own. It is distributed under the GNU General Public License.
I converted with ChatGPT and some trial and error.  I only converted the necessary pieces to get the Hellinger divergence.

The paper this is based on should be cited as follows:
"Influence Functions for Machine Learning: Nonparametric Estimators for Entropies, Divergences, and Mutual Informations",
by Kirthevasan Kandasamy, Akshay Krishnamurthy, Barnabas Poczos, Larry Wasserman, James Robins.

Their git repo is here: https://github.com/kirthevasank/if-estimators 

"""

import numpy as np
# from scipy.stats import gaussian_kde

from modules.influence_params import parse_two_distro_params
from modules.influence_getTwoDistroInfFunAvgs import get_two_distro_inf_fun_avgs

def get_influence_bounds(data0, data1, assume_even = False, handle_errors = "worst" ):

    estim, bwX, bwY = hellinger_divergence(data0, data1)
    BC = 1 - estim ### note that the Bhattacharyyya coefficient is 1- H^2

    if assume_even or  len(data0)== len(data1):
        upper = 1/2 * BC 
        lower = 1/2 - 1/2 * np.sqrt(1- BC**2)
    else:
        total = len(data0) + len(data1)
        P_c0 = len(data0) / total
        P_c1 = len(data1)/ total

        upper =    BC * np.sqrt(P_c0 *P_c1  )
        if BC > 1:
            if handle_errors == "worst": #thoeretical worst value for each 
                lower, upper = .5, .5
            elif handle_errors == "lower":
                lower =.5 
        else:
            lower = 1/2  - 1/2 * np.sqrt( 1- 4 *P_c0 *P_c1 *   (BC * BC))
   
    return lower, upper 




### This is the main funciton call to get the Hellinger distance SQUARED

def hellinger_divergence(X, Y, functional_params=None, params=None):
    if functional_params is None:
        functional_params = {'alpha': 0.5}

    # Call fAlphaGBeta function
    # estim1, asymp1, bwX, bwY = f_alpha_g_beta(X, Y, functional_params, params)
    estim1, bwX, bwY = f_alpha_g_beta(X, Y, functional_params, params)


    # Compute Hellinger Divergence
    estim = 1 - estim1


    return estim, bwX, bwY




def f_alpha_g_beta(X, Y, functional_params, params):
    # Estimate integral \int f^alpha g^beta where beta = 1-alpha
    params = parse_two_distro_params(params, X, Y)
    
    inf_fun_x = lambda dens_x_at_x, dens_y_at_x: f_alpha_g_beta_inf_fun_x(dens_x_at_x, dens_y_at_x, functional_params['alpha'])
    inf_fun_y = lambda dens_x_at_y, dens_y_at_y: f_alpha_g_beta_inf_fun_y(dens_x_at_y, dens_y_at_y, functional_params['alpha'])
    # taking this out 
    # asymp_var_fun = lambda dens_x_at_x, dens_x_at_y, dens_y_at_x, dens_y_at_y: f_alpha_g_beta_asymp_var(dens_x_at_x, dens_x_at_y, dens_y_at_x, dens_y_at_y, functional_params['alpha'])
    
    # Call get_two_distro_inf_fun_avgs function
    # estim, asymp_analysis, bwX, bwY = get_two_distro_inf_fun_avgs(X, Y, inf_fun_x, inf_fun_y, asymp_var_fun, params)
    estim, bwX, bwY = get_two_distro_inf_fun_avgs(X, Y, inf_fun_x, inf_fun_y, params)

    
    return estim, bwX, bwY # estim, asymp_analysis, bwX, bwY

def f_alpha_g_beta_inf_fun_x(dens_x_at_x, dens_y_at_x, alpha):
    # dens_x_at_x and dens_y_at_x are the densities of X and Y respectively (or their estimates) at the X points.
    inf_fun_vals = alpha * (dens_y_at_x / dens_x_at_x) ** (1 - alpha)
    return inf_fun_vals

def f_alpha_g_beta_inf_fun_y(dens_x_at_y, dens_y_at_y, alpha):
    # dens_x_at_y and dens_y_at_y are the densities of X and Y respectively (or their estimates) at the Y points.
    inf_fun_vals = (1 - alpha) * (dens_x_at_y / dens_y_at_y) ** alpha
    return inf_fun_vals


