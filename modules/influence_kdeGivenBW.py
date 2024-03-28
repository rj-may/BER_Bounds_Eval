import numpy as np

from modules.influence_kernel import kde_gauss_kernel, kde_legendre_kernel

def kde_given_bw(X, h, smoothness, params):
    # print(h)
    # Implements Kernel Density Estimator with kernels of order floor(smoothness)
    # for the given bandwidth. You should cross validate h externally.
    # Inputs
    #   X: the nxd data matrix
    #   h: bandwidth
    #   smoothness: If using a Gaussian Kernel this should be 'gaussian'. Otherwise
    #    specify the order of the legendre polynomial kernel.
    # Outputs
    #   kde: a function handle to estimate the density. kde takes in N points in a
    #     Nxd matrix and outputs an Nx1 vector.

    # prelims
    num_dims = X.shape[1]
    num_pts = X.shape[0]

    if 'doBoundaryCorrection' not in params:
        params['doBoundaryCorrection'] = True
    if 'estLowerBound' not in params:
        params['estLowerBound'] = 0
    if 'estUpperBound' not in params:
        params['estUpperBound'] = np.inf

    if not params['doBoundaryCorrection']:
        aug_X = X
    else:
        # First augment the dataset by mirroring the points close to the boundaries.
        aug_X = np.zeros((0, num_dims))
        # Our augmented space as 3^d regions. The centre region is the actual space
        # but all others are in the boundary. We iterate through them as follows
        for region_idx in range(3 ** num_dims):

            dim_regions = np.base_repr(region_idx, base=3)
            dim_regions = '0' * (num_dims - len(dim_regions)) + dim_regions
            # Now dim_regions is a string of dimRegions characters with each character
            # corrsponding to each dimension. If the character is 0, we look on the
            # lower boundary of the dimension and if 2 we look at the higher boundary
            # of the dimension.

            # Now check for points within h of the bounary and add them to the dataset
            to_replicate = np.ones((num_pts,))
            replic_X = X.copy()
            for d in range(num_dims):
                if dim_regions[d] == '0':
                    replic_X[:, d] = -replic_X[:, d]
                    to_replicate *= (X[:, d] < h).astype(int)
                elif dim_regions[d] == '2':
                    replic_X[:, d] = 2 - replic_X[:, d]
                    to_replicate *= (1 - (X[:, d] < h)).astype(int)
            replicated_pts = replic_X[to_replicate.astype(bool), :]
            aug_X = np.concatenate((aug_X, replicated_pts), axis=0)
            # Note that when dimRegions = '11...1', we will add the original X to augX

        num_aug_pts = aug_X.shape[0]

    # Now return the function handle
    return lambda arg: kde_iterative(arg, aug_X, h, smoothness, params, num_pts)


# A function which estimates the KDE at pts. We use this to construct the
# function handle which will be returned.
def kde_iterative(pts, aug_X, h, smoothness, params, num_X):

    num_pts = pts.shape[0]
    num_data = aug_X.shape[0]
    max_num_pts = max(1e7, num_data)
    pts_per_partition = min(num_pts, np.ceil(max_num_pts / num_data))

    ests = np.zeros((num_pts,))
    # Now iterate through each 'partition' and obtain the relevant kernels
    cum_num_pts = 0
    while cum_num_pts < num_pts:
        curr_num_pts = min(pts_per_partition, num_pts - cum_num_pts)
        if isinstance(smoothness, str) and smoothness.lower().startswith('gauss'):
            # print(h)
            K = kde_gauss_kernel(pts[cum_num_pts: cum_num_pts + curr_num_pts, :], aug_X, h)
        else:
            K = kde_legendre_kernel(pts[cum_num_pts: cum_num_pts + curr_num_pts, :], aug_X, h, smoothness)
        ests[cum_num_pts: cum_num_pts + curr_num_pts] = np.sum(K, axis=1) / num_X
        cum_num_pts += curr_num_pts

    # Now truncate those values below and above the bounds
    ests = np.maximum(ests, params['estLowerBound'])
    ests = np.minimum(ests, params['estUpperBound'])

    return ests

