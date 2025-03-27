
## Bayes Error Project  

#### Researcher:  Riley May 


I will be  running simulation-based comparisons of different methods of estimating Bayes Error Rate performance bounds under various conditions, including different sample sizes, numbers of dimensions, and distribution types. I will also be generating visualizations to summarize performance differences and reporting on the results of these simulations.



### Code description

I am going to try and provide up to date code descriptions of in the software manual
[here](https://github.com/rj-may/MS_Research/blob/master/Docs/Table_of_Contents.md) .


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
results= bounds_calculator(class0, class1, MATLAB = eng)

```
