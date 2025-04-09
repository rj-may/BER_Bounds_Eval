# Aribtrarily Tight Bounds using K-nn Density  Software Manual

This bound function uses the Bhattacharyya bounds and $k$-NN density estimates to get the Bhattacharyya bounds. 

This document is for the code found in `Bhatt_knn_func.py`. The bounds are calcualted using $k$-NN densisties. The code this functions relies on the k-nn density calculator function. Found [here](https://github.com/rj-may/BER_Bounds_Eval/edit/master/Docs/knn_density.md).


For a simple example of working with this funciton.


    from modules.Bhatt_knn_func import  Bhattacharyya_knn_bounds

    lower, upper =  Bhattacharyya_knn_bounds(data, data1, handle_error = "worst", k=0)

  The handle_errors term comes from high values of the Bhattacharyya coeefficient. The bounds shoudl never be greater than 0.5 . 


If one already has the density functions and their priors, the following can be used instead
     
     lower, upper = __calc_bha_knn_bounds(density0, density1, prior_c0, prior_c1)
