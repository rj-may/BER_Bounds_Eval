import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

# this function calculates the distance between two Distributions with the Bhattacharyya distance
def Bhattacharyya_dist(params1, params2, numpycheck = False):


    if numpycheck == False:
        test  = test_values(params1)  and test_values(params2)
        if test == False:
            return None

    mu1 = params1[0]
    covar1 = params1[1]

    mu2 = params2[0]
    covar2 = params2[1]

    Sigma = 1/2 * ( covar1 + covar2)

    Mahal_dist_2 = Mahalanobis_dist_sq(mu1, mu2, Sigma)
    
    dist =  1/8  * Mahal_dist_2 + 1/2 * np.log(det(Sigma) / np.sqrt((det(covar1 * covar2)  ) )   )

    return dist

def Bhattacharyya_bounds(params1, params2):

    test  = test_values(params1) and test_values(params2)
    if test == False: #if the test fails
        return None
    dist = Bhattacharyya_dist(params1, params2, numpycheck=True)

    error_rate = 1/2* ( 1 - np.sqrt(dist)) 
    
    bound1 = 1/2  - 1/2 * np.sqrt( 1- dist * dist)
    bound2 =  1/2 *   dist
    
    return bound1, bound2 


def  Mahalanobis_dist_sq(mu1, mu2, covar): #this returns the square of Mahalanobis distance
    mu_diff = mu1 - mu2

    return np.dot(mu_diff, np.matmul( inv(covar) , mu_diff)  )


def test_values(check):
    good = True
    for i in check:
        if type(i)!= np.ndarray:
            print( "Print ", i, "needs to be numpy array")
            good = False        
    return good

