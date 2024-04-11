''' only run this if you have sklearnex downloaded'''
# import sklearnex
# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.neighbors import NearestNeighbors
import numpy as np
import math


def Bhattacharyya_knn_bounds(data0, data1, k=0 , handle_errors = "worst"):
    if k == 0:
        k = knn_num_calc(len(data0), len(data0[0]))

    ## this is equivalent to the Bhattacharyya distance but we are using the knn densities
    BC =  __Bhattacharyya_coef_via_knn(data0, data1, k)

    # print(BC)

    # error_rate = 1/2* ( 1 - np.sqrt(dist)) 

    P_c0 = len(data0) /  ( len(data0) +  len(data1))
    P_c1 = len(data1) /( len(data0) +  len(data1))
    
    upper =    BC * np.sqrt(P_c0 *P_c1  )
    if BC > 1:
        if handle_errors == "worst": #thoeretical worst value for each 
            lower, upper = .5, .5
        elif handle_errors == "lower":
            lower =.5 
    else:
        lower = 1/2  - 1/2 * np.sqrt( 1- 4 *P_c0 *P_c1 *   (BC * BC))

    return lower, upper 


def __Bhattacharyya_coef_via_knn(data0, data1, k) :
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

    Px = 1/ len(X) #probability of x
    Pc0 =n0 / len(X) #probabilit of class 0
    Pc1 = n1 / len(X) # prob of class 1

    p0_x = density0 / (density0  + density1) ## P(c0| x)
    p1_x = density1 / (density0 + density1) ## P(c1 | x)

    # BC = np.sum(np.sqrt(px_0 *px_1 ))
    
    Px_c0 = p0_x * Px / Pc0 #P(x |c_0)
    Px_c1 = p1_x * Px / Pc1 #P(x |c_1)


    BC = np.sum(np.sqrt(Px_c0 * Px_c1))

    return BC




def __knn_density_calc(distances_matrix, k, p, n): # p is the dimension 
    vec = np.zeros(len(distances_matrix))

    for i in range(len(vec)):
        dist = distances_matrix[i][k-1]
        vol=  __calculate_volume(p, dist)
        vec[i] =  k /  ( n * vol  ) ### some people use k-1 for variance purposes
    return vec

def __calculate_volume(d, radius):
    return ((np.pi)**(d/2) ) / math.gamma((d/2) + 1) * (radius**d)



### used as a default /keyword parameter 
def knn_num_calc(N, d):# N is size of set and d is dimension
    mult = __multiplier(d)
    N_exp = N**(4/ (d+4))
    val=  int( mult * N_exp)
    if d <=2:
        print("This function doesn't work for dimension <3")
    return val


def __multiplier(n):
    num = n * (n + 2)**2 * math.gamma((n + 2) / 2)**(-4 / n) * ((n - 2) / n)**(2 + n / 2)
    denom = n**2 - 6 * n + 16
    return (num / denom)**(n / (n + 4))
