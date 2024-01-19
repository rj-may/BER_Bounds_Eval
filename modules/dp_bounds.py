import numpy as np
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import stats
import math


''' 
Distribution types this code works for

Multivariate normal: "mv_normal"

'''
accepted_distr = ["mv_normal"]

class dp_bounds:

    def __init__(self, dist_type, params0, params1, MC_Num, handle_errors = False):
        self.__distr_type = dist_type
        self.__params0 = params0
        self.__params1 = params1
        self.__MC_num = MC_Num
        self.__handle_errors = handle_errors


        if self.__distr_type not in accepted_distr:
            print("Not a programmed distribution " )
        
        self.__lower_bounds, self.__upper_bounds, self.__lower_stats, self.__upper_stats =  self.__simulate(self.__MC_num)

    def get_bounds(self):
        return [self.__lower_bounds, self.__upper_bounds]

    def get_bounds_stats(self):
        return [self.__lower_stats, self.__upper_stats]

    def __get_MC_num(self):
        return self.__MC_num
               
    def __get_distr_type(self):
        return self.__distr_type

    def __calc_bounds(self, up):            
        lower = 1/2 - 1/2 *math.sqrt(up) 
        upper = 1/2 - 1/2 * up 
        return lower, upper 
    
    def get_handle_errors(self):
        return self.__handle_errors


    def __get_FR(self, data1, data2, plot = False):    
        dataset = np.concatenate([data1, data2])

        FR_statistic =  0 
        # Calculate pairwise distances
        distances = distance.pdist(dataset)
        # Create a square distance matrix
        dist_matrix = distance.squareform(distances)
        # Create a minimum spanning tree
        mst = minimum_spanning_tree(dist_matrix)
        # Extract edges from the minimum spanning tree
        edges = np.array(np.where(mst.toarray() > 0)).T

        if plot == False:
            for edge in edges:
                if dataset[edge[0]] in data1 and dataset[edge[1]] in data1:
                    continue
                elif dataset[edge[0]] in data2 and dataset[edge[1]] in data2:
                    continue
                else:
                    FR_statistic  +=1

        else: #if we want to plot things
            fig = plt.figure(figsize = (7,11), )
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='blue', marker='o')
            ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='red', marker='o')

            for edge in edges:
                if dataset[edge[0]] in data1 and dataset[edge[1]] in data1:
                    color = 'blue'
                elif dataset[edge[0]] in data2 and dataset[edge[1]] in data2:
                    color = 'red'
                else:
                    color = 'purple'
                    FR_statistic  +=1

                ax.plot(dataset[edge, 0], dataset[edge, 1], dataset[edge, 2], c=color, )
            plot.show()
                
        return FR_statistic



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
            data1 =  np.random.multivariate_normal(mean1, covariance1, n0)
        
            FR  = self.__get_FR(data0, data1)

            Dp  = 1 - FR * (n0 + n1)/ (2 * n0 * n1)

            if n0 == n1:
                Up = Dp
            else:
                p = n0 /(n0 + n1)
                q = n1 / (n0 + n1)

                Up = 4 * p * q * Dp + (p-q)**2
                print("You are using distributions of size: ", n0, n1)

            if Up <= 0:
                print(self.get_handle_errors() )
                if self.get_handle_errors() == True:
                    print("Uh oh, you got a Up  = ", Up,  " We are omitting that from the data set")
                    continue
                elif self.get_handle_errors() == False:

                    print("Uh oh, you got a Up  = ", Up)
                    print("The FR statistics was ", FR)
                    print("We were on iteration  ", i )
                    self.__get_FR(data0, data1, True)
            

            lower, upper = self.__calc_bounds(Up)

            lower_bounds.append(lower)
            upper_bounds.append(upper)

        lower_stats = stats.describe(lower_bounds)
        upper_stats = stats.describe(upper_bounds)

        return lower_bounds, upper_bounds, lower_stats, upper_stats
        



            

            

