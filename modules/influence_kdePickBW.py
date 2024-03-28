# Here's the conversion of the `kdePickBW` MATLAB function to Python:

import numpy as np
from scipy.optimize import minimize_scalar

from modules.influence_kdeGivenBW import kde_given_bw


def kde_pick_bw(X, smoothness, params, bw_log_bounds=None):
    # This picks a bandwidth for the KDE. We use k-fold cross validation in the
    # range specified by bw_log_bounds.
    # If params.getKdeFuncH is True, then it also returns a function handle for the
    # kde with the optimal bandwidth.

    # prelims
    num_data = X.shape[0]
    num_dims = X.shape[1]
    USE_DIRECT = False

    # Shuffle the data
    shuffle_order = np.random.permutation(num_data)
    X = X[shuffle_order, :]

    # Obtain the Standard Deviation of X
    std_X = 1 if num_data == 1 else np.linalg.norm(np.std(X))

    # Set default parameter values
    if 'numPartsKFCV' not in params:
        params['numPartsKFCV'] = 5
    if 'numCandidates' not in params:
        params['numCandidates'] = 20
    if 'getKdeFuncH' not in params:
        params['getKdeFuncH'] = True
    if bw_log_bounds is None:
        if 'bwLogBounds' in params:
            bw_log_bounds = params['bwLogBounds']
        else:
            bw_log_bounds = np.log(np.array([1e-2, 10]) * std_X)
            bw_log_bounds[1] = min(bw_log_bounds[1], 1)

    if USE_DIRECT:
        # Use DiRect to Optimize over h
        diRectBounds = bw_log_bounds
        options = {'maxevals': params['numCandidates']}
        k_fold_func = lambda t: kde_k_fold_cv(t, X, smoothness, params)
        res = minimize_scalar(k_fold_func, bounds=diRectBounds, method='bounded', options=options)
        optBW = np.exp(res.x)
    else:
        # Use just ordinary KFold CV
        bw_candidates = np.linspace(bw_log_bounds[0], bw_log_bounds[1], params['numCandidates'])
        best_log_likl = -np.inf
        optBW = 0.2 * num_data ** (-4 / (4 + num_dims))
        for cand_iter in range(params['numCandidates']):
            curr_log_likl = kde_k_fold_cv(bw_candidates[cand_iter], X, smoothness, params)
            if curr_log_likl > best_log_likl:
                best_log_likl = curr_log_likl
                optBW = np.exp(bw_candidates[cand_iter])

    # Return a function handle
    if params['getKdeFuncH']:
        kde_func_h = kde_given_bw(X, optBW, smoothness, params)
    else:
        kde_func_h = None

    return optBW, kde_func_h


def kde_k_fold_cv(logBW, X, smoothness, params):
    h = np.exp(logBW)
    num_parts_kfcv = params['numPartsKFCV']
    log_likls = np.zeros(num_parts_kfcv)
    num_data = X.shape[0]
    num_dims = X.shape[1]

    for k_fold_iter in range(num_parts_kfcv):
        # Set the partition up
        test_start_idx = round((k_fold_iter) * num_data / num_parts_kfcv)
        test_end_idx = round((k_fold_iter + 1) * num_data / num_parts_kfcv)
        train_indices = np.concatenate([np.arange(0, test_start_idx), np.arange(test_end_idx, num_data)])
        test_indices = np.arange(test_start_idx, test_end_idx)
        num_test_data = test_end_idx - test_start_idx
        num_train_data = num_data - num_test_data
        # Separate Training and Validation sets
        Xtr = X[train_indices, :]
        Xte = X[test_indices, :]
        # Now Obtain the kde using Xtr
        kde_tr = kde_given_bw(Xtr, h, smoothness, params)
        # Compute Log Likelihood
        Pte = kde_tr(Xte)
        logPte = np.log(Pte)
        is_inf_logPte = np.isinf(logPte)
        # If fewer than 10% are infinities, then remove them
        if np.sum(is_inf_logPte) < 0.1 * num_test_data:
            logPte = logPte[~is_inf_logPte]
            log_likls[k_fold_iter] = np.mean(logPte)
        else:
            #       fprintf('%d/%d  =%0.4f, points had -inf loglikl. Quitting\n', ...
            #         sum(isInfLogPte), numTestData, sum(isInfLogPte)/numTestData);
            log_likls[k_fold_iter] = -np.inf
            break

    avg_log_likl = np.mean(log_likls)

    return avg_log_likl


# This Python conversion closely mirrors the functionality of the original MATLAB functions. Ensure that you use the correct data structures and keys in your Python code when using these functions.