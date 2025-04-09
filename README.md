
## Bayes Error Project  
Repo created as a part of MS research at Utah State University 

#### Researcher:  Riley May 


I ran simulation-based comparisons of different methods of estimating Bayes Error Rate performance bounds under various conditions, including different sample sizes, numbers of dimensions, and distribution types. I also be generated visualizations to summarize performance differences and reporting on the results of these simulations.




### Code description

Software manual for using the code in this repo is 
[here](https://github.com/rj-may/MS_Research/blob/master/Docs/Table_of_Contents.md) . Included here is documentation on the bounds_calculator function that is shown in the demo. 


The most useful code to other research is most likley the the code found within modules. That is the code used to calculate bounds and run the scenarios as presented in research. 



### demo
For a full demo see [https://github.com/rj-may/MS_Research/blob/master/demo.ipynb](https://github.com/rj-may/MS_Research/blob/master/demo.ipynb)


```
from modules.bounds_calculator import bounds_calculator
```

Simulate you data

```
import matlab.engine
eng = matlab.engine.start_matlab()
results= bounds_calculator(class0, class1, MATLAB = eng)

print(results)

eng.quit()
```

If you do not have access to matlab and the matlab engine,  that argument can be omitted

```
results= bounds_calculator(class0, class1, MATLAB = None)

```
