# Knn Density  Software Manual

For file knn_density.py


This function calculates the $k$ -Nearest neighbor density for given two classes of data. A value for $k$ is recommended to be provided, but if it is not available $k$ 
will be calculated using a normal assumption. 



`get_knn_densities` returns two density vectors that are the density of the data at each point across the two datasets.  If the two classes are far apart, then at each point that is a location in 
class 0 will have a low value for the density at the location of the class 1 points. 




    from modules.knn_densities import get_knn_densities, knn_num_calc
    p0, p1 get_knn_densities(data0, data1, k=0) 


If someone wants to just get a good value for $k$ based off a normal assumption , then the following funciton can be used where $N$ is the number of samples and $D$ is the number of dimensiosn. 

    k = knn_num_calc(N,D)
