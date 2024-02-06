import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

### this is the function to use
# it accepts parameters based on the calculated distributions
def Bhattacharyya_bounds(params1, params2, handle_errors = "worst"):

    test  = test_values(params1) and test_values(params2)
    if test == False: #if the test fails
        return None
    dist = __Bhattacharyya_dist(params1, params2, numpycheck=True)

    # error_rate = 1/2* ( 1 - np.sqrt(dist)) 
    BC  = np.exp( -1 * dist ) # claculate the Bhattacharyya coefficient
    
    upper =  1/2 *   BC
    if BC >=1:
        if handle_errors == "worst": #thoeretical worst value for each 
            lower, upper = .5, .5
        elif handle_errors == "lower":
            lower =.5
    else:
        lower = 1/2  - 1/2 * np.sqrt( 1- BC * BC)

    return lower, upper 



# this function calculates the distance between two Distributions with the Bhattacharyya distance
def __Bhattacharyya_dist(params1, params2, numpycheck = False):


    if numpycheck == False:
        test  = test_values(params1)  and test_values(params2)
        if test == False:
            return None

    mu1 = params1[0]
    covar1 = params1[1]

    mu2 = params2[0]
    covar2 = params2[1]

    Sigma = 1/2 * ( covar1 + covar2)

    Mahal_dist_2 = __Mahalanobis_dist_sq(mu1, mu2, Sigma)
    
    dist =  1/8  * Mahal_dist_2 + 1/2 * np.log(det(Sigma) / np.sqrt((det(covar1 * covar2)  ) )   )

    return dist




def  __Mahalanobis_dist_sq(mu1, mu2, covar): #this returns the square of Mahalanobis distance
    mu_diff = mu1 - mu2

    return np.dot(mu_diff, np.matmul( inv(covar) , mu_diff)  )


def test_values(check):
    good = True
    for i in check:
        if type(i)!= np.ndarray:
            print( "Print ", i, "needs to be numpy array")
            good = False        
    return good

