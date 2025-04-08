import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

def get_knn_densities(data0, data1, k) :
    if k == 0:
        k = knn_num_calc(len(data0), len(data0[0]))
    
    X = np.concatenate([data0, data1])# merge two class data sets to get our X space

    p = len(data0[0]) ## the dimension of the data sets
    n0 = len(data0)
    n1 = len(data1)

    # Fit k-Nearest Neighbors model and get densitities for data set 1
    knn = NearestNeighbors(n_neighbors=k, algorithm = 'auto')
    knn.fit(data0)
    distances, indices = knn.kneighbors(X) # get distance to the  1,2,... kth nearest neighbor across the space x
    density0 = __knn_density_calc(distances, k, p, n0) ## calculate density based off the distances, k, dim, and sample size

    # Fit k-Nearest Neighbors model and get densitities for data set 2
    knn = NearestNeighbors(n_neighbors=k, algorithm= 'auto')
    knn.fit(data1)
    distances, indices = knn.kneighbors(X)    
    density1 = __knn_density_calc(distances, k, p, n1)

    # print(sum(density0), sum(density1))
    ### VERSION 1

    # Px = 1/ len(X) #probability of x
    # Pc0 =n0 / len(X) #probabilit of class 0
    # Pc1 = n1 / len(X) # prob of class 1

    # p0_x = density0 / (density0  + density1) ## P(c0| x)
    # p1_x = density1 / (density0 + density1) ## P(c1 | x)

    
    # Px_c0 = p0_x * Px / Pc0 #P(x |c_0)
    # Px_c1 = p1_x * Px / Pc1 #P(x |c_1)
    # BC = np.sum(np.sqrt(Px_c0 * Px_c1))

    # VERSION 2
    p0 = density0 / sum(density0)
    p1 = density1 / sum(density1)

    return p0, p1


def __knn_density_calc(distances_matrix, k, p, n): # p is the dimension 
    vec = np.zeros(len(distances_matrix))

    for i in range(len(vec)):
        dist = distances_matrix[i][k-1]
        vol=  __calculate_volume(p, dist)
        # print(p, dist, vol)
        # print("N", n)
        # print("K", k)
        # print("vol", vol)
        vec[i] =  (k-1) /  ( n * vol  ) ###  people use k-1 for variance purposes
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
