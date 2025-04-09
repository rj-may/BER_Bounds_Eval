# EnDive  Calculator Software Manual


Included in this repository is code from Kevin Moon's repo. Found here [https://github.com/KevinMoonLab/EnDive)(https://github.com/KevinMoonLab/EnDive)]

The following matlab files found in the modules are not from my own creation. I claim no ownership, but are provided as a convenience. 
  - EnDive.m
  - calculateWeightsKDE.m
  - calculateWeightsKDE_cvx.m
  - Note this doesn't include two other files necessary for full utility of the EnDive stimator. 


The code was used in the followin way in this repo. 

For a very simple example of getting the $D_p$ divergence the following can be done. 

      import matlab.engine
      eng = matlab.engine.start_matlab()
      eng.addpath('modules', nargout=0)
      
      dp = eng.EnDive(data0, data1, 'type', "DP", 'quiet', 'kernel', kernel, 'est', 2, nargout=1)
      eng.quit()

Note the followin function combines this approach with the influcence bounds to get out the bounds with only one call to the Matlab . 

https://github.com/rj-may/BER_Bounds_Eval/blob/master/modules/matlab_calc.m
