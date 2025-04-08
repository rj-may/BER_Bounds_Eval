# Bhattacharyya Bounds Calculator Software Manual
 ```Bhattacharyya_func.py```

Introduction

The python module has the following two functions that are intended to be used from it. 
- `get_Bhattacharyya_bounds`
- `get_Maha_upper`

  ` get_Bhattacharyya_bounds ` 
This function calculates the bounds with the assumption that the data comes from two multivariate normal distributions. The function calcualtes the mean and covariance matrices for the two data classes using nummpy and returns the Bhattacharyyya bounds. 

The function `get_Maha_upper` calculates the Mahalanobis_upper bound based off the data. 

Make sure to have the necessary libraries installed:    ``` numpy ```




Usage

## Example Usage
    lower_bound, upper_bound = Bhattacharyya_bounds(data0, data1, assume_even=Fals )

    upper_Maha =  get_Maha_upper(data0, data1, assume_even = False):

    


Alternatively if you arleady know the mean dand covaraince matrix the funciton `Bhattacharyya_bounds` is also available.   
Parameter types
- parameters1 and 2- list with [ (n by 1 list), (n by n numpy array), (integer value)]




