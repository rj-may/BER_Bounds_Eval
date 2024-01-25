import numpy as np
from scipy import stats
import math
from numpy.linalg import inv
from numpy.linalg import det


''' 
Distribution types this code works for

Multivariate normal: "mv_normal"

'''
accepted_distr = ["mv_normal"]


class Bhattacharyya_bounds:

    def __init__(self, dist_type, params0, params1, MC_Num):
        self.__distr_type = dist_type
        self.__params0 = params0
        self.__params1 = params1
        self.__MC_num = MC_Num


        if self.__distr_type not in accepted_distr:
            print("Not a programmed distribution " )
        
        self.__lower_bounds, self.__upper_bounds, self.__lower_stats, self.__upper_stats =  self.__simulate(self.__MC_num)


    #these two funcitons are main two returnables for 
    def get_bounds(self): 
        return [self.__lower_bounds, self.__upper_bounds]

    def get_bounds_stats(self):
        return [self.__lower_stats, self.__upper_stats]


    def __get_MC_num(self):
        return self.__MC_num
               
    def __get_distr_type(self):
        return self.__distr_type



   
    def __obs_params(self, data):
        mean = np.mean(data, axis=0)# this getting it by the column I believe this should work for [[x y z] [x y z ]] data
        covar = np.cov(data, rowvar= False)
        return [mean, covar]

        
    # this function calculates the distance between two Distributions with the Bhattacharyya distance
    def ___Bhattacharyya_dist(self, params1, params2):
        mu1 = params1[0]
        covar1 = params1[1]
        mu2 = params2[0]
        covar2 = params2[1]

        Sigma = 1/2 * ( covar1 + covar2)

        Mahal_dist_2 = self.__Mahalanobis_dist_sq(mu1, mu2, Sigma)
        
        dist =  1/8  * Mahal_dist_2 + 1/2 * np.log(det(Sigma) / np.sqrt((det(covar1 * covar2)  ) )   )
        return dist


    def  __Mahalanobis_dist_sq(self, mu1, mu2, covar): #this returns the square of Mahalanobis distance
        mu_diff = mu1 - mu2

        return np.dot(mu_diff, np.matmul( inv(covar) , mu_diff)  )


    def __Bhattacharyya_bounds(self, params1, params2):

        dist = self.___Bhattacharyya_dist(params1, params2) #calculate the distance

        # error_rate = 1/2* ( 1 - np.sqrt(dist)) 
        BC  = np.exp( -1 * dist ) # claculate the Bhattacharyya coefficient
        bound1 = 1/2  - 1/2 * np.sqrt( 1- BC * BC)
        bound2 =  1/2 *   BC
        
        return bound1, bound2 


    ### this is the main looping function
    def __simulate(self, MC_iter):
        MC_iter = MC_iter
        lower_bounds =[]
        upper_bounds = []

        if self.__get_distr_type() == "mv_normal":    
            # params should be (mean1, covariance1, n0) where means is  1 x n list and covar is n x n
            mean0, covariance0, n0 = self.__params0
            mean1, covariance1, n1 = self.__params1

        for i in range(MC_iter):
            
            data0 =  np.random.multivariate_normal(mean0, covariance0, n0)
            data1 =  np.random.multivariate_normal(mean1, covariance1, n1)


            sim_params0 = self.__obs_params(data0)
            sim_params1 = self.__obs_params(data1)

            lower, upper = self.__Bhattacharyya_bounds(sim_params0, sim_params1)
        
           


            lower_bounds.append(lower)
            upper_bounds.append(upper)

        lower_stats = stats.describe(lower_bounds)
        upper_stats = stats.describe(upper_bounds)

        return lower_bounds, upper_bounds, lower_stats, upper_stats
        '''
        stats.describe documentaiton

    Number of observations (length of data along axis). When ‘omit’ is chosen as nan_policy, the length along each axis slice is counted separately.

minmax: tuple of ndarrays or floats
    Minimum and maximum value of a along the given axis. 

    meanndarray or float     Arithmetic mean of a along the given axis.


variancendarray or float
    Unbiased variance of a along the given axis; denominator is number of observations minus one.

skewnessndarray or float
    Skewness of a along the given axis, based on moment calculations with denominator equal to the number of observations, i.e. no degrees of freedom correction.

kurtosisndarray or float
    Kurtosis (Fisher) of a along the given axis. The kurtosis is normalized so that it is zero for the normal distribution. No degrees of freedom are used.

        '''
        



            

            

