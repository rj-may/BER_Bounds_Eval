import numpy as np
from sklearn.mixture import GaussianMixture


class data_gen_gauss_mix:

    def __init__(self, params0,  params1,  boundary = None ):  # default argument at the end
        self.__means0 = np.array(params0['means'])  # for the Gaussian mixture EXAMPLE  np.array([[0], [3], [6]])
        self.__stds0 = np.array(params0['covariances'])  # for the Gaussian mixture for multidimensional it is the covariance EXAMLE = np.array([[1], [0.5], [2]])

        self.__params1 = params1 ### these must be for a mulitdimensional gaussian 

        self.__dist1 = distribution(np.random.multivariate_normal, self.__params1)

        self.__dimension = len(params1['mean']) ### This determines the dimensionality of the samplin
        
        self.__gaussian_counts = len(self.__means0)


        self.__boundary = boundary


        # Initialize the GaussianMixture model with pre-defined parameters
        gmm = GaussianMixture(n_components = self.__gaussian_counts)

        # Set the parameters manually for the 1D case
        gmm.means_ = self.__means0
        gmm.covariances_ = self.__stds0  

        gmm.weights_ = np.ones(self.__gaussian_counts ) / ( self.__gaussian_counts)

        self.__gmm = gmm


        
    def sample(self, sample_size):
            # vector = np.random.binomial( self.__gaussian_counts - 1, 1/ self.__gaussian_counts,  size =sample_size)
            # # print(vector)

            # _ , counts = np.unique(vector, return_counts=True)
            # weights  = counts / sample_size

            # print(weights)

            gmm = self.__gmm

            # gmm.weights_ = weights

            samples, _ = gmm.sample(n_samples=sample_size)


            if self.__dimension == 1:
                # print(type(self.__dist1))
                return samples, self.__dist1.sample(sample_size)

            else:
                class0 = []
                class0.append(samples)


                more_samples = np.random.multivariate_normal(mean = np.zeros(self.__dimension -1), cov = np.identity(self.__dimension -1), size = sample_size )
                class0.append(more_samples)

                combined = [np.concatenate((samples[i], more_samples[i])) for i in range(len(samples))]
                
                # print(np.shape(samples), np.shape(more_samples))

                # print(samples, more_samples)

                return  np.array(combined), self.__dist1.sample(sample_size)
            

    
    def __str__(self):
        return "The Gaussian Mixture is " + str(self.__means0) + str(self.__stds0) 
    

    def has_boundary(self):
        return self.__boundary != None

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




class distribution:

    def __init__(self, func, params):
        self.__func = func
        self.__params = params

    def sample(self, num ):
        return self.__func(size = num, **self.__params)