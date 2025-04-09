import numpy as np
import math

from scipy.spatial import cKDTree

def get_knn_densities(data0, data1, k=0):
    if k == 0:
        k = knn_num_calc(len(data0), len(data0[0]))
    
    X = np.concatenate([data0, data1])
    p = len(data0[0]) #dimension
    n0 = len(data0)
    n1 = len(data1)

    # Use KDTree on data0
    tree0 = cKDTree(data0)
    distances0, _ = tree0.query(X, k=k)
    density0 = __knn_density_calc_scipy(distances0, k, p, n0)

    # Use KDTree on data1
    tree1 = cKDTree(data1)
    distances1, _ = tree1.query(X, k=k)
    density1 = __knn_density_calc_scipy(distances1, k, p, n1)

    p0 = density0 / sum(density0)
    p1 = density1 / sum(density1)

    return p0, p1

def __knn_density_calc_scipy(distances, k, p, n):
    # Ensure distances is 2D
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
    vec = np.zeros(len(distances))
    for i in range(len(vec)):
        dist = distances[i][k - 1]
        vol = __calculate_volume(p, dist)
        vec[i] = (k - 1) / (n * vol)
    return vec




def __calculate_volume(d, radius):
    return ((np.pi)**(d/2) ) / math.gamma((d/2) + 1) * (radius**d)

# '''
# The following formula uses formula for n- dimesional sphere 
# $$ p_k(x)  =\frac{k}{n} \frac{1}{ \frac{\pi^{p/2}}{\Gamma(p/2+1)}  \|x-x_k \|^p}.$$
# '''



def knn_num_calc(N, d, supress = False):# N is size of set and d is dimension
    if d < 3 :
        print("This function doesn't work for dimension <3")
    mult = __multiplier(d)
    N_exp = N**(4/ (d+4))
    val=  round( mult * N_exp)
    if val < 3:
        if not supress:
            print(val,  " was calculated for k. 3 is chosen for k for variance purposes.")
        val = 3
    return val


def __multiplier(n):
    num = n * (n + 2)**2 * math.gamma((n + 2) / 2)**(-4 / n) * ((n - 2) / n)**(2 + n / 2)
    denom = n**2 - 6 * n + 16
    return (num / denom)**(n / (n + 4))
