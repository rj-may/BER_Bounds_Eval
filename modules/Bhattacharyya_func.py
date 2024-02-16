import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

### this is the function to use
# it accepts parameters based on the calculated distributions
#class probability assumes distributions of equal class sizes


### This really only works for class that 

def Bhattacharyya_bounds(params0, params1, class_prob = [1/2, 1/2], handle_errors = "worst"):

    test  = test_values(params0) and test_values(params1)
    if test == False: #if the test fails
        return None
    if class_prob[0] + class_prob[1] != 1:
        print("Why are the class distributions not summing to 1?")

    dist = __Bhattacharyya_dist(params0, params1, class_prob, numpycheck=True)

    # error_rate = 1/2* ( 1 - np.sqrt(dist)) 
    BC  = np.exp( -1 * dist ) # calculate the Bhattacharyya coefficient
    # print(BC)
    P_c0 = class_prob[0]
    P_c1 = class_prob[1]
    
    upper =    BC * np.sqrt(P_c0 *P_c1  )
    if BC > 1:
        if handle_errors == "worst": #thoeretical worst value for each 
            lower, upper = .5, .5
        elif handle_errors == "lower":
            lower =.5 
    else:
        lower = 1/2  - 1/2 * np.sqrt( 1- 4 *P_c0 *P_c1 *   (BC * BC))

    return lower, upper 



# this function calculates the distance between two Distributions with the Bhattacharyya distance
def __Bhattacharyya_dist(params0, params1, class_prob = [0.5, 0.5],  numpycheck = False):

    if numpycheck == True:
        test  = test_values(params0)  and test_values(params1)
        if test == False:
            return None

    mu0 = params0[0]
    covar0 = params0[1]

    mu1 = params1[0]
    covar1 = params1[1]

    Sigma = class_prob[0]* covar0 +  class_prob[1] * covar1

    Mahal_dist_2 = __Mahalanobis_dist_sq(mu0, mu1, Sigma)
    
    dist =  1/8  * Mahal_dist_2 + 1/2 * np.log(det(Sigma) / np.sqrt((det(covar0 * covar1)  ) )   )

    return dist


def Mahalanobis_upper( params0, params1, class_prob = [.5, .5]):
    mu0 = params0[0]
    covar0 = params0[1]

    mu1 = params1[0]
    covar1 = params1[1]

    P_c0 = class_prob[0]
    P_c1 = class_prob[1]

    num = 2 * P_c0 * P_c1
    
    sigma_for_Mah = class_prob[0] * covar0 + class_prob[1] * covar1
    Mahal_dist_2 = __Mahalanobis_dist_sq( mu0, mu1, sigma_for_Mah)
    denom = 1 + P_c0 * P_c1 * Mahal_dist_2

    return (num/denom)



def  __Mahalanobis_dist_sq(mu0, mu1, covar): #this returns the square of Mahalanobis distance
    ### the covariance matrix should be a weighted  covariance matrix of each distribution  
    mu_diff = mu0 - mu1

    return np.dot(mu_diff, np.matmul( inv(covar) , mu_diff)  )


def test_values(check):
    good = True
    for i in check:
        if type(i)!= np.ndarray:
            print( "Print ", i, "needs to be numpy array")
            good = False        
    return good

