# import numpy as np


def parse_two_distro_params(params, X, Y):
    # X, Y are data from densities f and g respectively.
    # params is a struct with other parameters (viz. whether to data split or not)
    n, num_dims = X.shape
    m = Y.shape[1]
    params = parse_common_params(params, num_dims, min(m, n))
    return params

def parse_common_params(params, num_dims, n):
    # params is a struct with some parameters for the estimation.
    # (viz. whether to data split or not)
    # num_dims is the number of Dimensions
    if params is None:
        params = {}

    # # Whether to do Asymptotic Analysis
    # if 'doAsympAnalysis' not in params:
    #     params['doAsympAnalysis'] = False

    # The smoothness of the function for the KDE
    if 'smoothness' not in params:
        params['smoothness'] = 'gauss'

    # Number of partitions to split the data into
    if 'numPartitions' not in params:
        params['numPartitions'] = 1  # by default, do not partition the data.
    if isinstance(params['numPartitions'], str) and params['numPartitions'] == 'loo':
        params['numPartitions'] = n

    # Number of partitions to average over
    if 'numAvgPartitions' not in params:
        if 'averageAll' in params and not params['averageAll']:
            params['numAvgPartitions'] = 1
        else:
            params['numAvgPartitions'] = params['numPartitions']

    # Some parameters for Kernel Density estimation
    if 'doBoundaryCorrection' not in params:
        params['doBoundaryCorrection'] = False

    if 'estLowerBound' not in params:
        params['estLowerBound'] = 1e-5

    if 'estUpperBound' not in params:
        params['estUpperBound'] = float('inf')

    return params
