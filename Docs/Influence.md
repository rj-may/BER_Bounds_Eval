# EnDive  Calculator Software Manual


Included in this repository is code from someone else's research on Nonparametric Estimation using Influence Estimators. 
Found here [git link](https://github.com/kirthevasank/if-estimators)

The following matlab files found in the modules are not from my own creation. I claim no ownership, but are provided as a convenience. 
  - distSquared.m
  - fAlphaBGBeta.m
  - getTwoDistroInfFunAvgs.m
  - hellingerDivergence.m
  - kdeGivenBW.m
  - kdePickBW.m
  - parseTwoDistroParams.m
  - Note this doesn't include  other files necessary for full utility of the influence estimator

The code was used in the followin way in this repo. 

For a very simple example of getting the $D_p$ divergence the following can be done. 

      import matlab.engine
      eng = matlab.engine.start_matlab()
      eng.addpath('modules', nargout=0)
      
      estim_Hellinger = eng.hellingerDivergence(data0, data1,[], [],  nargout= 1)
      BC = 1 - estim #get Bhattacharyya coefficient 
      
      eng.quit()

Note the following function combines this approach with the EnDive bounds to get out the bounds with only one call to the Matlab . 

https://github.com/rj-may/BER_Bounds_Eval/blob/master/modules/matlab_calc.m
