''' 
this is the master class that can calculate multiple bounds
'''
from modules.dp_func import get_bounds_dp as calc_bounds_dp
from modules.tight_knn_func import get_tight_bounds_knn as calc_tight_bounds_knn
from modules.Bhattacharyya_func import Bhattacharyya_bounds as calc_Bhattacharyya_bounds
from modules.Bhattacharyya_func import Mahalanobis_upper as calc_Mahalanobis_upper
from modules.Bhatt_knn_func import Bhattacharyya_knn_bounds as calc_Bhatt_knn_bounds
import numpy as np
import math
import concurrent.futures  # Add this line to import the concurrent module

accepted_distr = ["mv_normal"]

accepted_bound_types =  ["dp","tight", "Bhattacharyya", "Bhatt_knn", "Mahalanobis"]

class bounds_class:

    def __init__(self, distr_type, params0, params1, MC_num=500, threads =2, bound_types = accepted_bound_types, dp_handle_errors ="worst", Bha_handle_errors= "worst", Bha_knn_handle_errors ="worst", alpha_tight =50 , k_nn =0 ):
        self.__distr_type = distr_type
        self.__params0 = params0
        self.__params1 = params1
        self.__MC_num = MC_num
        self.__threads= threads
        self.__bound_types= bound_types
        self.__dp_handle_errors = dp_handle_errors
        self.__Bha_handle_errrors = Bha_handle_errors
        self.__Bha_knn_handle_errors = Bha_knn_handle_errors
        self.__tight_bounds_alpha = alpha_tight # for the aribitrarilty tight bound density type. 
        self.__tight_bounds_knn_num =  k_nn

        if self.__distr_type not in accepted_distr:
            print("Not a programmed distribution~sincerely multi bound class " )

        for i in self.__bound_types:
            if i not in accepted_bound_types:
                print("Not a an accepted bound type")
        
        self.__lower_bounds_dp = []
        self.__upper_bounds_dp = []
        self.__lower_bounds_Bha =  []
        self.__upper_bounds_Bha =  []
        self.__lower_bounds_Bha_knn =  []
        self.__upper_bounds_Bha_knn =  []
        self.__lower_bounds_tight = []
        self.__upper_bounds_tight = []
        self.__upper_bounds_Maha = []
        
      
        self.__parallel_simulation(self.__MC_num, self.__threads) #wild fun 

    def __len__(self):
        return self.__MC_num

    def __call__(self):
        values_dict = {"Dp": np.mean(self.get_bounds_dp(), axis = 1), "Bha": np.mean(self.get_bounds_Bha(), axis = 1),
                        "Bha_knn": np.mean(self.get_bounds_Bha_knn(), axis =1), "tight": np.mean(self.get_bounds_tight(), axis =1),
                        "Maha": np.mean(self.get_upper_Maha()) }
        return values_dict


    def get_bounds_dp(self):
        return self.__lower_bounds_dp, self.__upper_bounds_dp
    def get_bounds_Bha(self):
        return self.__lower_bounds_Bha, self.__upper_bounds_Bha
    def get_bounds_tight(self):
        return self.__lower_bounds_tight, self.__upper_bounds_tight

    def get_bounds_Bha_knn(self):
        return self.__lower_bounds_Bha_knn, self.__upper_bounds_Bha_knn
    def get_upper_Maha(self):
        return self.__upper_bounds_Maha

    def __str__(self):
        params = self.__get_params()
        a = "The distribution type is: " +  str(self.__get_distr_type()) +  "with " + str(self.get_MC_num()) +  " Monte Carlo Iterations"
        b = "Distribution 0 is " +  str(params[0]) 
        c = "Distribution 1 is " + str(params[1])
        return a + "\n "+ "\n " + b + "\n" + c
    
    def get_info(self):
        return [self.__distr_type, self.__get_params()[0], self.__get_params()[1], self.get_MC_num()]


    def get_MC_num(self):
        return self.__MC_num

    def __get_bound_types(self):
        return self.__bound_types

    def __get_params(self):
        return [self.__params0, self.__params1]

    def __get_distr_type(self):
        return self.__distr_type

    def get_handle_errors_dp(self):
        return self.__dp_handle_errors
    
    def get_handle_errors_Bha(self):
        return self.__Bha_handle_errrors

    def get_handle_errors_Bha_knn(self):
        return self.__Bha_knn_handle_errors

    def __obs_params(self, data):
        mean = np.mean(data, axis=0)# this getting it by the column I believe this should work for [[x y z] [x y z ]] data
        covar = np.cov(data, rowvar= False)
        return [mean, covar]


    ##this code is written so there is no issues with multiple functions accessing the same list in __parallel_simulation
    def __simulate_for_parallel(self, MC_iter):
        MC_iter = MC_iter


        if self.__get_distr_type() == "mv_normal":    
            # params should be (mean1, covariance1, n0) where means is  1 x n list and covar is n x n
            mean0, covariance0, n0 = self.__params0
            mean1, covariance1, n1 = self.__params1
        
        lower_bounds_dp, upper_bounds_dp, lower_bounds_tight, upper_bounds_tight, lower_bounds_Bha, upper_bounds_Bha = [], [], [], [], [], []
        lower_bounds_Bha_knn , upper_bounds_Bha_knn, upper_bounds_Maha = [], [], []
        
        for i in range(MC_iter):
            
            data0 =  np.random.multivariate_normal(mean0, covariance0, n0)
            data1 =  np.random.multivariate_normal(mean1, covariance1, n1)
            
            sim_params0 = self.__obs_params(data0)
            sim_params1 = self.__obs_params(data1)
            if "dp" in self.__get_bound_types():
                dp_l, dp_u = calc_bounds_dp(data0, data1, handle_errors = self.get_handle_errors_dp())
                lower_bounds_dp.append(dp_l)
                upper_bounds_dp.append(dp_u)

            if "tight" in self.__get_bound_types():
                l, u = calc_tight_bounds_knn(data0, data1, alpha=self.__tight_bounds_alpha, k=self.__tight_bounds_knn_num)
                lower_bounds_tight.append(l)
                upper_bounds_tight.append(u)

            if "Bhattacharyya" in self.__get_bound_types():
                # sim_params0 = self.__obs_params(data0)
                # sim_params1 = self.__obs_params(data1)

                l, u = calc_Bhattacharyya_bounds(sim_params0, sim_params1, handle_errors = self.get_handle_errors_Bha())
                lower_bounds_Bha.append(l)
                upper_bounds_Bha.append(u)
            
            if "Bhatt_knn" in self.__get_bound_types():
                a, b = calc_Bhatt_knn_bounds(data0, data1, k = self.__tight_bounds_knn_num, handle_errors= self.get_handle_errors_Bha_knn())
                lower_bounds_Bha_knn.append(a)
                upper_bounds_Bha_knn.append(b)
            if "Mahalanobis" in  self.__get_bound_types():
                # sim_params0 = self.__obs_params(data0)
                # sim_params1 = self.__obs_params(data1)
                a =calc_Mahalanobis_upper(sim_params0, sim_params1, [n0 /(n0+n1), n1/(n0+n1)] )
                upper_bounds_Maha.append(a)

        return lower_bounds_dp, upper_bounds_dp, lower_bounds_tight, upper_bounds_tight, lower_bounds_Bha, upper_bounds_Bha, lower_bounds_Bha_knn, upper_bounds_Bha_knn, upper_bounds_Maha


    def __parallel_simulation(self, MC_iter, num_threads):
        MC_iter_per_thread = MC_iter // num_threads

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.__simulate_for_parallel, MC_iter_per_thread) for _ in range(num_threads)]

            for future in concurrent.futures.as_completed(futures):
                # print(future.result())
                lower_bounds_dp, upper_bounds_dp, lower_bounds_tight, upper_bounds_tight, lower_bounds_Bha, upper_bounds_Bha, lower_bounds_Bha_knn, upper_bounds_Bha_knn, upper_bounds_Maha  = future.result()
                self.__lower_bounds_dp.extend(lower_bounds_dp)
                self.__upper_bounds_dp.extend(upper_bounds_dp)
                self.__lower_bounds_tight.extend(lower_bounds_tight)
                self.__upper_bounds_tight.extend(upper_bounds_tight)
                self.__lower_bounds_Bha.extend(lower_bounds_Bha)
                self.__upper_bounds_Bha.extend(upper_bounds_Bha)
                self.__lower_bounds_Bha_knn.extend(lower_bounds_Bha_knn)
                self.__upper_bounds_Bha_knn.extend(upper_bounds_Bha_knn)
                self.__upper_bounds_Maha.extend(upper_bounds_Maha)