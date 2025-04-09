# Bound Calculator Software brief how -to



This function uses calculates 3 types of bounds  from various methods. To read about any of them see the [Table of Contents](https://github.com/rj-may/BER_Bounds_Eval/blob/master/Docs/Table_of_Contents.md)
  - Bhattacharrya via parametric normal assumption
  - Bhattacharyya via k-Nearest Neighbor density
  - Bhattacharyya via Influence Functions *
  - Dp Bounds via Minimum Spanning Tree
  - Dp bounds via ensemble divergence estimates *
  - Aribtrarily Tight Bounds via  k-Nearest Neighbor Estimates


The methods labeled with asterick * require the use of a maltlab engine. If you set `MATLAB =None ` those bounds will not be included in the output dictionary. 
    
    import matlab.engine
    eng = matlab.engine.start_matlab()
    results= bounds_calculator(class0, class1, MATLAB = eng)
    
    print(results)
    
    eng.quit()

To use the bounds calculator it will require the use of sci-py, numpy, matlab.engine and scikit-learn*(optional see below) to get every bound.


## output
The results are ouput as a dicitonary. One usage of this function would be to calculate the bounds for lots of different simulations, and store the results dictionary in 
a list. Then use Pandas or something else to make a dataframe from it. 

    {'dp_lower': 0.0669872981077807,
     'dp_upper': 0.125,
     'Bhattacharyya_lower': 0.04322723735465056,
     'Bhattacharyya_upper': 0.2033682455678252,
     'tight_lower': 0.11186294736504052,
     'tight_upper': 0.1161965890695767,
     'Bha_knn_lower': 0.08191701108624444,
     'Bha_knn_upper': 0.2742382438336064,
     'enDive_lower': 0.0549588701770074,
     'enDive_upper': 0.1038767855317485,
     'influence_lower': 0.0368680550826741,
     'influence_upper': 0.1884377923801248}



## Further use and Tmer
    bounds_calculator(data0, data1, k_nn=0, alpha_tight=50, kernel='uniform', MATLAB= None, Timer = False, sckitlearn= False)

  - k-NN is the number to use for the k_nn density. If you don't give one or give 0, it will be calculated for you. 
  - alpha is for the arbitrarily tight bounds. 
  - kernel is for the Ensemble Divergence methods (EnDive). 
  - Timer if set to true this changes the output. THis gives the time to calculate the bounds as well as the result .
      - results, times = bounds_calculator(data0, data1,  Timer = True)
  - Only set sckitlearn equal to True if you are paranoid about replicating my results. This just uses scikitlearn to calculate the nearest neighobrs which was how everything was cacluated. But Scikit-Learn is built on scipy, so having scikitlearn just uses scipy directly and is one less package. .

