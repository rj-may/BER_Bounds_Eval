# DP Bounds Calculator Software Manual



``` dp_bounds.py``` 

Introduction

The DP Bounds Calculator is a Python tool designed to calculate lower and upper bounds for the using a DP measure  (DP) between two multivariate normal distributions. 


Ensure you have the necessary libraries installed:


    pip install numpy scipy matplotlib

Usage
Initializing the Calculator

To use the calculator, create an instance of the dp_bounds class, specifying the distribution type, parameters for the first distribution (params0), parameters for the second distribution (params1), the number of Monte Carlo iterations (MC_Num), error handling method (handle_errors), and whether to suppress messages (suppress_message).

Parameter types:

- dist_type: String, accepted values: "mv_normal"
- params1 and params2: List with [ (n by 1 list), (n by n numpy array), (integer value)]
- MC_Num: Number of Monte Carlo iterations - Integer
- handle_errors: Error handling method - String, options: 'omit', 'no', 'worst' Default is 'worst'
- suppress_message: Boolean, whether to suppress 'omit' messages - Boolean



### Example initialization


    from modules.dp_bounds import dp_bounds
    
    calculator = dp_bounds("mv_normal", [mean0, covariance0, n0], [mean1, covariance1, n1], MC_Num, 'worst', False)

Obtaining Bounds

Use the get_bounds() method to retrieve the computed lower and upper bounds.


    # Example usage
    
    bounds = calculator.get_bounds()
    print("Lower Bound:", bounds[0])
    print("Upper Bound:", bounds[1])

Obtaining Bounds Statistics

Use the get_bounds_stats() method to retrieve descriptive statistics of the lower and upper bounds. The statistics are from the SciPy stats module.

  
  
    # Example usage
    bounds_stats = calculator.get_bounds_stats()
    print("Lower Bounds Statistics:", bounds_stats[0])
    print("Upper Bounds Statistics:", bounds_stats[1])

Supported Distribution Type

The calculator currently supports the multivariate normal distribution ("mv_normal").

Handling Errors

The ```handle_errors``` parameter controls how the calculator handles errors. Options:

    'omit': Omit errors from the data set and continue. DO NOT USE. CREATES A NEW DATA DISTRIBUTION
    'no': Print statistics and attempt to provide a plot.
    'worst': Provide both the upper and lower bounds with a value of 0.5.

## Example

Here's a complete example using sample data:



    # Sample data
    mean0 = [0, 0]
    covariance0 = np.identity(2)
    n0 = 100
    
    mean1 = [1, 0]
    covariance1 = np.identity(2)
    n1 = 100
    
    MC_Num = 1000
    
    # Initialize calculator
    calculator = dp_bounds("mv_normal", [mean0, covariance0, n0], [mean1, covariance1, n1], MC_Num, 'worst')
    
    # Obtain  bounds
    bounds = calculator.get_bounds()

    
    # Obtain and print bounds statistics
    bounds_stats = calculator.get_bounds_stats()
    print("Lower Bounds Statistics:", bounds_stats[0])
    print("Upper Bounds Statistics:", bounds_stats[1])


## Threaded example

This is taken from the ```mse_plot.ipynb```

This was created form the original file but I used ChatGPT to produce the threaded code. 
It seems to run best when '''threads =2`` for smaller sample sizes, but increasing the thread size to 4 may be beneficial for large samples. 

Make sure the ```MC_num``` is divisible by the ```threads=number```



    sample_sizes = np.logspace(2, 3.3, 7 , endpoint = True, dtype = int)


    from modules.dp_bounds_threaded import dp_bounds as dp_bounds2

    dp_lst = []
           
    
    
    for i in sample_sizes:
        start = time.time()
    
        n0, n1 = i, i
        params1  = [mean1, covariance1, n0]
        params2  = [mean2, covariance2, n1]
        
        dp_class = dp_bounds2('mv_normal', params1, params2, MC_num, threads = 2, handle_errors= 'worst')
    #     dp_class = dp_bounds('mv_normal', params1, params2, MC_num, handle_errors= 'worst')
    
    
        dp_lst.append(dp_class)
