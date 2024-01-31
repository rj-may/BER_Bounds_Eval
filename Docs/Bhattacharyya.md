Bhattacharyya Bounds Calculator Software Manual
Introduction

The Bhattacharyya Bounds Calculator is a Python tool designed to compute lower and upper bounds for the Bhattacharyya coefficient between two multivariate normal distributions. This calculator is particularly useful in statistical analysis and pattern recognition when assessing the similarity between two sets of data.
Installation

Make sure to have the necessary libraries installed:  
    numpy scipy


Usage
Initializing the Calculator

To use the calculator, create an instance of the Bhattacharyya_bounds class, specifying the distribution type, parameters for the first distribution (params0), parameters for the second distribution (params1), and the number of Monte Carlo iterations (MC_Num).



    # Example initialization
    calculator = Bhattacharyya_bounds("mv_normal", [mean0, covariance0, n0], [mean1, covariance1, n1], MC_Num)

Obtaining Bounds

Use the get_bounds() method to retrieve the computed lower and upper bounds.
    
    # Example usage
    
    bounds = calculator.get_bounds()
    print("Lower Bound:", bounds[0])
    print("Upper Bound:", bounds[1])


Obtaining Bounds Statistics

Use the get_bounds_stats() method to retrieve descriptive statistics of the lower and upper bounds. The statisticcs are from scipy stats


    # Example usage
    
    bounds_stats = calculator.get_bounds_stats()
    print("Lower Bounds Statistics:", bounds_stats[0])
    print("Upper Bounds Statistics:", bounds_stats[1])

Supported Distribution Type

The calculator currently supports the multivariate normal distribution ("mv_normal").

Here's a complete example using sample data:
    
    # Sample data
    mean0 = [0, 0]
    covariance0 = np.identity(2)
    n0 = 100
    
    mean1 = [1, 0]
    mean1 = np.array
    covariance1 = np.identity(2)
    n1 = 100
    
    MC_Num = 1000
    
    # Initialize calculator
    calculator = Bhattacharyya_bounds("mv_normal", [mean0, covariance0, n0], [mean1, covariance1, n1], MC_Num)
    
    # Obtain and print bounds
    bounds = calculator.get_bounds()
    print("Lower Bound:", bounds[0])
    print("Upper Bound:", bounds[1])
    
    # Obtain and print bounds statistics
    bounds_stats = calculator.get_bounds_stats()
    print("Lower Bounds Statistics:", bounds_stats[0])
    print("Upper Bounds Statistics:", bounds_stats[1])

Conclusion

The Bhattacharyya Bounds Calculator provides a convenient way to assess the similarity between multivariate normal distributions.
