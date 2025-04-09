# DP Bounds Calculator Software Manual



``` dp_bounds.py``` 

Introduction

The DP Bounds function is a Python tool designed to calculate lower and upper bounds for the using a divergence measure defined [here](https://ieeexplore.ieee.org/document/7254229) (DP).
It relies on scipy to calculate a minimum spanning tree. 

The function to use is written as 
`get_bounds_dp(data0, data1, handle_errors = "worst"):`

Data0 is class 0 and data1 is for class1. The handle errors is for if there is instability and it predicts that the Dp divergence is greater than 1 it will return that both the upper and lower bounds are 0.5

Also, the function `get_dp_bounds(data0, data1, handle_errors = "worst"):` produces the same result. 



Ensure you have the necessary libraries installed:


    pip install numpy scipy 

Usage
Initializing the Calculator




    # Sample data
    from modules.dp_func import get_dp_bounds

    dim = 3
    
    diff = 2.56
    
    mean1 = np.zeros(dim)
    covariance1 = np.identity(dim)
    mean2 = np.zeros(dim)
    mean2[0] = diff # set the difference between the two means
    covariance2= np.identity(dim)
            
    samples = 500
    
    data0 = np.random.multivariate_normal(mean1, covariance1, samples)
    data1 = np.random.multivariate_normal(mean2, covariance2, samples)

    
    
    
    # Initialize calculator
    lower , upper = get_dp_bounds(data0, data1)
    
