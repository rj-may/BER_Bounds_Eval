import numpy as np

class data_gen:

    def __init__(self, func0, func1, params0, params1,  dimensions = 3 ):
        self.__params0 = params0
        self.__params1 = params1
        
        ### create two classes with certain parameters that you can smample from
        self.__dist0 = distribution(func0, self.__params0)
        self.__dist1 = distribution(func1, self.__params1)
        self.__dim = dimensions

        if type(self.__dim) != int:
            print("Invalid distributution. Not an integer for dimenions")


    def sample(self, size ):

        if self.__dim== 1:
            return self.__dist0.sample(size), self.__dist1.sample(size)
        else:
            class0 = []
            class1 = []
            a,b =  self.__dist0.sample(size), self.__dist1.sample(size)

            class0.append(a)
            class1.append(b)

            for i in range(self.__dim -1):
                a = np.random.uniform(-1, 1, size)
                b = np.random.uniform(-1, 1, size)
                class0.append(a)
                class1.append(b)  
            return np.array(class0).T, np.array(class1).T
    
    def __len__(self):
        return self.__dim
    
    def __str__(self):
        print(self.__params0)
        print(self.__params1)
    
        


class distribution:

    def __init__(self, func, params):
        self.__func = func
        self.__params = params

    def sample(self, num ):
        return self.__func(size = num, **self.__params)