import numpy as np
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
import math


def get_bounds_dp(data0, data1):
    FR = __get_FR(data0, data1, False)
    n0 = len(data0)
    n1 = len(data1)

    Dp = 1 - FR * (n0 + n1)/ (2 * n0 * n1)
    if n0 == n1:
        return __calc_bounds(Dp)
    else:
        p = n0 / (n0 +n1)
        q = n1/ (n0 +n1)
        up = 4 * p * q * Dp  + (p-q)^2
        return __calc_bounds(up)
    


def __get_FR(data0, data1, plot = False):    
        dataset = np.concatenate([data0, data1])

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
                if dataset[edge[0]] in data0 and dataset[edge[1]] in data0:
                    continue
                elif dataset[edge[0]] in data1 and dataset[edge[1]] in data1:
                    continue
                else:
                    FR_statistic  +=1

        else: #if we want to plot things
            fig = plt.figure(figsize = (7,11), )
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data0[:, 0], data0[:, 1], data0[:, 2], c='blue', marker='o')
            ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='red', marker='o')

            for edge in edges:
                if dataset[edge[0]] in data0 and dataset[edge[1]] in data0:
                    color = 'blue'
                elif dataset[edge[0]] in data1 and dataset[edge[1]] in data1:
                    color = 'red'
                else:
                    color = 'purple'
                    FR_statistic  +=1

                ax.plot(dataset[edge, 0], dataset[edge, 1], dataset[edge, 2], c=color)
            plot.show()
                
        return FR_statistic

    
def __calc_bounds(up):
    upper = 1/2 - 1/2 * up 
    if up> 0:
        lower = 1/2 - 1/2 *math.sqrt(up) 
    else:
        lower =.5
    return lower, upper 

def analyze(data0, data1):

    dataset = np.concatenate([data0, data1])

    FR_statistic =  0 

    # Calculate pairwise distances

    distances = distance.pdist(dataset)

    # Create a square distance matrix
    dist_matrix = distance.squareform(distances)

    # Create a minimum spanning tree
    mst = minimum_spanning_tree(dist_matrix)

    # Extract edges from the minimum spanning tree
    edges = np.array(np.where(mst.toarray() > 0)).T

    # Plot the dataset and the minimum spanning tree
    fig = plt.figure(figsize = (7,11), )
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c='black', marker='o')

    for edge in edges:
        if dataset[edge[0]] in data0 and dataset[edge[1]] in data0:
            color = 'blue'
#             marker = 'blue'
        elif dataset[edge[0]] in data1 and dataset[edge[1]] in data1:
            color = 'red'
#             marker = "r"
        else:
            color = 'cyan'
#             marker = 'g'
            FR_statistic  +=1

        ax.plot(dataset[edge, 0], dataset[edge, 1], dataset[edge, 2], c=color, )

    plt.show()
    print("Frieman-Rafsky statistic = ",  FR_statistic )