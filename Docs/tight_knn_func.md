# Aribtrarily Tight Bounds using K-nn Density  Software Manual

This bound function comes from the work in a paper called Arbitrarily Tight Upper and Lower Bounds on the Bayesian Probability of Error found https://doi.org/10.1109/34.476017. 


This document is for the code found in `tight_knn_func.py`. The bounds are calcualted using $k$-NN densisties. The code this functions  relies on the k-nn density calculator function. Found [here](https://github.com/rj-may/BER_Bounds_Eval/edit/master/Docs/knn_density.md).


For a simple example of working with this funciton.


    from modules.tight_kn_func import get_tight_bounds_knn

    lower, upper = get_tight_bounds_knn(data, data1, alpha = 50, k=0)

  The values of alpha come from the paper and is assumed to be 50. The value for k is optional. If one is not provided one will be calculated using a normal assumption. 


If one already has the density functions and their priors, the following can be used instead
     
     lower, upper, __calc_tight_bounds_via_knn_density(density0, density1, prior_c0, prior_c1,  alpha)

