import numpy as np

class data_gen:

    def __init__(self, func0, func1, params0, params1,  dimensions = 3, boundary = None ):
        self.__params0 = params0
        self.__params1 = params1
        
        ### create two classes with certain parameters that you can smample from
        self.__dist0 = distribution(func0, self.__params0)
        self.__dist1 = distribution(func1, self.__params1)
        self.__dim = dimensions

        self.__boundary = boundary

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
        
    
    def get_est_BER(self, class0, class1):
        if self.__boundary == None:
            return "There is no specified boundary for this dataset"
        else:
            a = class0[:, 0 ]
            b = class1[:, 0]
            
            if isinstance(self.__boundary, (int, float, complex)):
                crossing = self.__boundary
                count0 = 0
                for val in b:
                    if val > crossing:
                        count0 += 1
                count0 = min(len(a) - count0, count0)

                count1 =  0 
                for val in a:
                    if val < crossing :
                        count1 += 1
                count1 = min(len(b) - count1, count1)

                return (count0 + count1) / (len(a) + len(b))

            elif len(self.__boundary) >= 2:
                lower_b = self.__boundary[0]
                upper_b = self.__boundary[len(self.__boundary)-1]

                count0 = 0
                for val in a:
                    if lower_b< val and  val < upper_b:
                        count0 += 1
                count0 = min(len(a) - count0, count0)

                count1 = 0
                for val in b:
                    if lower_b< val and  val < upper_b:
                        count1 += 1
                count1 = min(len(b) - count1, count1)
            
                return (count0 + count1) / (len(a) + len(b))
            else:
                return "unknown scenario"


        
    def __len__(self):
        return self.__dim
    
    def __str__(self):
        print(self.__params0)
        print(self.__params1)
    
    def has_boundary(self):
        return self.__boundary != None
    
        


class distribution:

    def __init__(self, func, params):
        self.__func = func
        self.__params = params

    def sample(self, num ):
        return self.__func(size = num, **self.__params)