import numpy as np

#data generator class for a multivariate distribution. 

class data_gen_multivariate:

    def __init__(self, func0, func1, params0, params1):
        self.__params0 = params0
        self.__params1 = params1
        
        ### create two classes with certain parameters that you can smample from
        self.__dist0 = distribution(func0, self.__params0)
        self.__dist1 = distribution(func1, self.__params1)




    def sample(self, size ):

        return self.__dist0.sample(size), self.__dist1.sample(size)

    
    # def __len__(self):
    #     return 
    
    def __str__(self):
        print(self.__params0)
        print(self.__params1)
    
        


class distribution:

    def __init__(self, func, params):
        self.__func = func
        self.__params = params

    def sample(self, num ):
        return self.__func(size = num, **self.__params)