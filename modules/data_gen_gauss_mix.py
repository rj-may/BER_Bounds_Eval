import numpy as np
from sklearn.mixture import GaussianMixture


class data_gen_gauss_mix:

    def __init__(self, params0,  params1):  # default argument at the end
        self.__means0 = np.array(params0['means'])  # for the Gaussian mixture EXAMPLE  np.array([[0], [3], [6]])
        self.__stds0 = np.array(params0['covariances'])  # for the Gaussian mixture for multidimensional it is the covariance EXAMLE = np.array([[1], [0.5], [2]])

        self.__params1 = params1 ### these must be for a mulitdimensional gaussian 

        self.__dist1 = distribution(np.random.multivariate_normal, self.__params1)

        self.__dimension = len(params1['mean']) ### This determines the dimensionality of the samplin
        
        self.__gaussian_counts = len(self.__means0)


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
      



class distribution:

    def __init__(self, func, params):
        self.__func = func
        self.__params = params

    def sample(self, num ):
        return self.__func(size = num, **self.__params)