import numpy as np
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
import math

def analyze(data1, data2):

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

    # Plot the dataset and the minimum spanning tree
    fig = plt.figure(figsize = (7,11), )
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c='black', marker='o')

    for edge in edges:
        if dataset[edge[0]] in data1 and dataset[edge[1]] in data1:
            color = 'blue'
#             marker = 'blue'
        elif dataset[edge[0]] in data2 and dataset[edge[1]] in data2:
            color = 'red'
#             marker = "r"
        else:
            color = 'cyan'
#             marker = 'g'
            FR_statistic  +=1

        ax.plot(dataset[edge, 0], dataset[edge, 1], dataset[edge, 2], c=color, )

    plt.show()
    print("Frieman-Rafsky statistic = ",  FR_statistic )