"""
This is a sub-file of the influence code 

Disclaimer:
This Python file is based on code from a library that is not my own. It is distributed under the GNU General Public License.

The paper this is based on should be cited as follows:
"Influence Functions for Machine Learning: Nonparametric Estimators for Entropies, Divergences, and Mutual Informations",
by Kirthevasan Kandasamy, Akshay Krishnamurthy, Barnabas Poczos, Larry Wasserman, James Robins.

Their git repo is here: https://github.com/kirthevasank/if-estimators 

"""

import numpy as np

def kde_gauss_kernel(X, Y, h):
    # Returns the Kernel Matrix for a Gaussian Kernel of bandwidth h.
    # X is an nxd matrix. K is an nxn matrix.
    # If Y is nonempty then returns the gaussian kernel for XxY
    # h is a column vector of size the dimension of the space - i.e size(X, 1).

    # Prelims
    d = X.shape[1]  # dimensions

    # if h.ndim == 1:
    #     h = h.reshape(-1, 1)  # if you get a row vector

    if np.isscalar(h):
        h = h * np.ones((d, 1))

    if Y is None:
        Y = X

    D2 = dist_squared(X, Y, h)
    K = 1 / (np.sqrt(2 * np.pi) ** d * np.prod(h)) * np.exp(-D2 / 2)

    return K


def dist_squared(X, Y, h=1):
    # distanceSquared: Calculates squared distance between two sets of points.
    # If h is provided, this scales each dimension by the bandwidth

    nX, dX = X.shape
    nY, dY = Y.shape
    if dX != dY:
        raise ValueError('Dimensions of X, Y do not match')
    


    # X_scaled = X / h.reshape(1, -1)
    # Y_scaled = Y / h.reshape(1, -1)
    # print(h)
    X_scaled = X / np.reshape(h, (1, -1))
    Y_scaled = Y /  np.reshape(h, (1, -1))


    D2 = np.sum(X_scaled ** 2, axis=1, keepdims=True) + \
         np.sum(Y_scaled ** 2, axis=1, keepdims=True).T - \
         2 * X_scaled.dot(Y_scaled.T)

    # Rounding errors occasionally cause negative entries in D2
    D2[D2 < 0] = 0

    return D2



def kde_legendre_kernel(X, C, h, order):
    # Returns the value of the kernel evaluated at the points X centred at C and
    # with bandwidth h.
    # Inputs
    # X : nxd data matrix
    # C : mxd centre matrix. If empty is initialized to zero(1, d)
    # h : the bandwidth of the kernel
    # order : order of the kernel
    # Ouputs
    # K : The nxm kernel matrix where K(i,j) = k(X(i,:), C(j,:))
    # Warning: make sure mxn < 1e6 to avoid crashing

    # Prelims
    num_dims = X.shape[1]

    if C is None:
        C = np.zeros((1, num_dims))

    num_data = X.shape[0]
    num_centres = C.shape[0]

    K = np.ones((num_data, num_centres))
    for d in range(num_dims):
        K *= kernel_1d(X[:, d], C[:, d], h, order)

    return K


def kernel_1d(x, c, h, order):
    # Same as above but now x and c are 1 dimensional (d=1)

    num_centres = c.shape[0]
    # u is a numData x numCentres matrix, u_ij = (x_i - c_j)/h
    u = np.subtract.outer(x, c) / h

    ret = np.zeros_like(u)
    for m in range(0, order + 1, 2):
        # only need to iterate through even m since legPoly(0,m) = 0 for m odd
        ret += leg_poly(0, m) * leg_poly(u, m)
    # Finally check if u is within the domain of the kernel and divide by h.
    ret *= (np.abs(u) < 1).astype(float) / h
    return ret


def leg_poly(x, order):
    # Legendre polynomial evaluation
    if order == 0:
        return np.ones_like(x)
    elif order == 2:
        return 0.5 * (3 * x ** 2 - 1)
    elif order == 4:
        return (35 * x ** 4 - 30 * x ** 2 + 3) / 8
    else:
        raise NotImplementedError("Legendre polynomial of order {} is not implemented.".format(order))
