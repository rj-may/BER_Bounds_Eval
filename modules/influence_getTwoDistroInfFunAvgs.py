
"""
This is a sub-file of the influence code 

Disclaimer:
This Python file is based on code from a library that is not my own. It is distributed under the GNU General Public License.

The paper this is based on should be cited as follows:
"Influence Functions for Machine Learning: Nonparametric Estimators for Entropies, Divergences, and Mutual Informations",
by Kirthevasan Kandasamy, Akshay Krishnamurthy, Barnabas Poczos, Larry Wasserman, James Robins.

Their git repo is here: https://github.com/kirthevasank/if-estimators 

"""

# Your code starts here


import numpy as np
from scipy.stats import norm

from modules.influence_kdePickBW import kde_pick_bw

from modules.influence_kdeGivenBW import kde_given_bw


def get_two_distro_inf_fun_avgs(X, Y, inf_fun_x, inf_fun_y, params):
    # This function will sum the influence functions over the partitions

    n = X.shape[0]
    m = Y.shape[0]
    num_partitions = params['numPartitions']
    num_avg_partitions = params['numAvgPartitions']

    inf_fun_x_part_terms = np.zeros(num_avg_partitions)
    inf_fun_y_part_terms = np.zeros(num_avg_partitions)
    # asymp_var_x_part_terms = np.zeros(num_avg_partitions) #don't need
    # asymp_var_y_part_terms = np.zeros(num_avg_partitions)
    part_weights_x = np.zeros(num_avg_partitions)
    part_weights_y = np.zeros(num_avg_partitions)

    for k in range(num_avg_partitions):
        X_den, X_est = get_den_est_samples(X, num_partitions, k)
        Y_den, Y_est = get_den_est_samples(Y, num_partitions, k)

        if k == 0:
            # For X data
            if 'bandwidthX' not in params or params['bandwidthX'] is None:
                # print("here1")
                bwX, kdeFuncH_X = kde_pick_bw(X_den, params['smoothness'], params)
                # print("here3", bwX) ### I guess we need to understand why is  kde PICK bw returning the second paramter, does it have a use MATLAB behaves oddly
            else:
                bwX = params['bandwidthX'](X_den)
            # For Y data
            if 'bandwidthY' not in params or params['bandwidthY'] is None:
                bwY, kdeFuncH_Y = kde_pick_bw(Y_den, params['smoothness'], params)
            else:
                bwY = params['bandwidthY'](Y_den)

        dens_est_X = kde_given_bw(X_den, bwX, params['smoothness'], params)
        dens_est_Y = kde_given_bw(Y_den, bwY, params['smoothness'], params)

        dens_X_at_X = dens_est_X(X_est)
        dens_X_at_Y = dens_est_X(Y_est)
        dens_Y_at_X = dens_est_Y(X_est)
        dens_Y_at_Y = dens_est_Y(Y_est)

        inf_fun_x_part_terms[k] = np.sum(inf_fun_x(dens_X_at_X, dens_Y_at_X))
        inf_fun_y_part_terms[k] = np.sum(inf_fun_y(dens_X_at_Y, dens_Y_at_Y))
        part_weights_x[k] = X_est.shape[0]
        part_weights_y[k] = Y_est.shape[0]


    avg = np.sum(inf_fun_x_part_terms) / np.sum(part_weights_x) + np.sum(inf_fun_y_part_terms) / np.sum(part_weights_y)



    return avg, bwX, bwY


def get_den_est_samples(X, num_partitions, curr_idx):
    # X is the entire dataset. num_partitions is the total number of partitions and
    # curr_idx is the current partition Idx

    if num_partitions == 1:
        X_den = X
        X_est = X
    else:
        n = X.shape[0]
        start_idx = round((curr_idx - 1) * n / num_partitions + 1)
        end_idx = round(curr_idx * n / num_partitions)
        est_idxs = list(range(start_idx - 1, end_idx))  # Adjust indices to Python 0-indexing
        den_idxs = list(range(0, start_idx - 1)) + list(range(end_idx, n))
        X_den = X[den_idxs, :]
        X_est = X[est_idxs, :]

    return X_den, X_est




