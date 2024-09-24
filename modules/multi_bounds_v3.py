''' 
this is the master class that can calculate multiple bounds
'''
from modules.dp_func import get_bounds_dp as calc_bounds_dp
from modules.Bhattacharyya_func import get_Bhattacharyya_bounds as calc_Bhattacharyya_bounds
from modules.Bhattacharyya_func import get_Maha_upper as calc_Mahalanobis_upper

from modules.tight_knn_func import __calc_tight_bounds_via_knn_density as calc_tight_bounds_knn
from modules.Bhatt_knn_func import __calc_bha_knn_bounds as calc_Bhatt_knn_bounds
from modules.knn_density import get_knn_densities as calc_knn_densities

# from modules.influence import get_influence_bounds as calc_influence_bounds

import matlab.engine


import numpy as np
import math
import concurrent.futures  # Add this line to import the concurrent module

accepted_distr = ["mv_normal"]

accepted_bound_types =  ["dp","tight", "Bhattacharyya", "Bhatt_knn", "Mahalanobis", "influence", "enDive"]

error_dict ={"dp_handle_errors" :"worst", "Bha_handle_errors": "worst", "Bha_knn_handle_errors":"worst", "influence_handle_errors": "worst"}

class bounds_class:

    def __init__(self, data_generator,  sample_size = 500, MC_num=500, threads =2, bound_types = accepted_bound_types, error_dict = error_dict, alpha_tight =50,  k_nn =0,kernel= 'uniform' ):
        self.__data_generator = data_generator
        self.__MC_num = MC_num
        self.__sample_size = sample_size
        self.__threads= threads
        self.__bound_types= bound_types
        self.__dp_handle_errors = error_dict["dp_handle_errors"]
        self.__Bha_handle_errrors = error_dict["Bha_handle_errors"]
        self.__Bha_knn_handle_errors = error_dict["Bha_knn_handle_errors"]
        self.__influence_handle_errors= error_dict["influence_handle_errors"]
        self.__tight_bounds_alpha = alpha_tight # for the aribitrarilty tight bound density type. 
        self.__knn_num =  k_nn
        self.__kernel = kernel



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
        self.__lower_bounds_inf = []
        self.__upper_bounds_inf = []
        self.__lower_bounds_enDive = []
        self.__upper_bounds_enDive = []

        self.__parallel_simulation(self.__MC_num, self.__threads) #wild fun 

    def __len__(self):
        return self.__MC_num

    def __call__(self):
        dp_bounds_l, dp_bounds_u =   np.mean(self.get_bounds_dp(), axis = 1)
        bha_bounds_l, bha_bounds_u = np.mean(self.get_bounds_Bha(), axis = 1)
        bha_knn_bounds_l, bha_knn_bounds_u =  np.mean(self.get_bounds_Bha_knn(), axis =1)
        tight_bounds_l, tight_bounds_u  =  np.mean(self.get_bounds_tight(), axis =1)
        inf_l, inf_u  =  np.mean(self.get_inf_bounds(), axis =1)
        enDive_l, enDive_u = np.mean(self.get_Bounds_enDive(), axis =1)

        values_dict = {
            "Dp_lower": dp_bounds_l,
            "Dp_upper": dp_bounds_u,
            "Bha_lower": bha_bounds_l,
            "Bha_upper": bha_bounds_u,
            "Bha_knn_lower": bha_knn_bounds_l,
            "Bha_knn_upper": bha_knn_bounds_u, 
            "tight_lower": tight_bounds_l,
            "tight_upper": tight_bounds_u,
            "Maha_upper": np.mean(self.get_upper_Maha()),
            "inf_lower": inf_l,
            "inf_upper": inf_u,
            "enDive_lower": enDive_l,
            "enDive_upper": enDive_u
            }

        return values_dict

    def validity(self, BER):
        true = np.ones(self.__MC_num)* BER
        dp_bounds_l, dp_bounds_u =   self.get_bounds_dp()
        bha_bounds_l, bha_bounds_u = self.get_bounds_Bha()
        bha_knn_bounds_l, bha_knn_bounds_u =  self.get_bounds_Bha_knn()
        tight_bounds_l, tight_bounds_u  =  self.get_bounds_tight()
        inf_l, inf_u = self.get_inf_bounds()
        enDive_l, enDive_u = self.get_Bounds_enDive()
        Maha_upper = self.get_upper_Maha()

        values_dict = {
            "Dp_lower":  np.sum((true - dp_bounds_l)>0) / self.__MC_num if dp_bounds_l else np.nan,
            "Dp_upper": np.sum((dp_bounds_u - true)>0) / self.__MC_num if dp_bounds_u else np.nan,
            "Bha_lower":  np.sum((true - bha_bounds_l)>0) / self.__MC_num if bha_bounds_l else np.nan ,
            "Bha_upper":  np.sum((bha_bounds_u - true)>0) / self.__MC_num if bha_bounds_u else np.nan,
            "Bha_knn_lower":  np.sum((true - bha_knn_bounds_l)>0) / self.__MC_num if bha_knn_bounds_l else np.nan,
            "Bha_knn_upper":  np.sum((bha_knn_bounds_u - true)>0) / self.__MC_num if bha_knn_bounds_u else np.nan, 
            "tight_lower":  np.sum((true - tight_bounds_l)>0) / self.__MC_num  if tight_bounds_l else np.nan,
            "tight_upper":  np.sum((tight_bounds_u - true)>0) / self.__MC_num if tight_bounds_u else np.nan,
            "Maha_upper":  np.sum((Maha_upper - true)>0) / self.__MC_num if Maha_upper else np.nan ,
            "inf_lower": np.sum((true - inf_l )>0) / self.__MC_num  if inf_l else np.nan,
            "inf_upper": np.sum((inf_u - true)>0) / self.__MC_num if inf_u else np.nan,
            "enDive_lower": np.sum((true - enDive_l )>0) / self.__MC_num if enDive_l else np.nan,
            "enDive_upper": np.sum((enDive_u - true)>0) / self.__MC_num if enDive_u else np.nan,
            "Dp": np.sum(np.logical_and((true - dp_bounds_l) > 0, (dp_bounds_u - true) > 0)) / self.__MC_num if dp_bounds_l else np.nan,
            "Bha":  np.sum(np.logical_and((true - bha_bounds_l) > 0, (bha_bounds_u - true) > 0)) / self.__MC_num if bha_bounds_l else np.nan,
            "Bha_knn": np.sum(np.logical_and((true - bha_knn_bounds_l) > 0, (bha_knn_bounds_u - true) > 0)) / self.__MC_num if bha_knn_bounds_l else np.nan,
            "tight": np.sum(np.logical_and((true - tight_bounds_l) > 0, (tight_bounds_u - true) > 0)) / self.__MC_num if tight_bounds_l else np.nan,
            "inf": np.sum(np.logical_and((true - inf_l) > 0, (inf_u - true) > 0)) / self.__MC_num if inf_l else np.nan,
            "enDive": np.sum(np.logical_and((true - enDive_l) > 0, (enDive_u - true) > 0)) / self.__MC_num if enDive_l else np.nan
            }

        return values_dict
    

    
    def bound_width(self):
        dp_bounds_l, dp_bounds_u =   self.get_bounds_dp()
        bha_bounds_l, bha_bounds_u = self.get_bounds_Bha()
        bha_knn_bounds_l, bha_knn_bounds_u =  self.get_bounds_Bha_knn()
        tight_bounds_l, tight_bounds_u  =  self.get_bounds_tight()
        inf_l, inf_u = self.get_inf_bounds()
        enDive_l, enDive_u = self.get_Bounds_enDive()
        Maha_upper = self.get_upper_Maha()


        values_dict = {

            "Dp": np.mean(np.subtract(dp_bounds_u, dp_bounds_l)) if dp_bounds_l else np.nan,
            "Bha": np.mean(np.subtract(bha_bounds_u, bha_bounds_l)) if bha_bounds_l else np.nan,
            "Bha_knn": np.mean(np.subtract(bha_knn_bounds_u, bha_knn_bounds_l)) if bha_knn_bounds_l else np.nan,
            "tight": np.mean(np.subtract(tight_bounds_u, tight_bounds_l)) if tight_bounds_l else np.nan,
            "inf": np.mean(np.subtract(inf_u, inf_l)) if inf_l else np.nan,
            "enDive": np.mean(np.subtract(enDive_u, enDive_l)) if enDive_l else np.nan
            }

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
    def get_inf_bounds(self):
        return self.__lower_bounds_inf, self.__upper_bounds_inf
    def get_Bounds_enDive(self):
        return self.__lower_bounds_enDive, self.__upper_bounds_enDive

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

    # def __obs_params(self, data):
    #     mean = np.mean(data, axis=0)# this getting it by the column I believe this should work for [[x y z] [x y z ]] data
    #     covar = np.cov(data, rowvar= False)
    #     return [mean, covar]


    ##this code is written so there is no issues with multiple functions accessing the same list in __parallel_simulation
    def __simulate_for_parallel(self, MC_iter):
        MC_iter = MC_iter

        
        eng = matlab.engine.start_matlab()
        eng.cd(r'modules', nargout=0)

        
        lower_bounds_dp, upper_bounds_dp, lower_bounds_tight, upper_bounds_tight, lower_bounds_Bha, upper_bounds_Bha = [], [], [], [], [], []
        lower_bounds_Bha_knn , upper_bounds_Bha_knn, upper_bounds_Maha = [], [], []
        lower_bounds_inf, upper_bounds_inf, lower_bounds_enDive, upper_bounds_enDive = [], [], [], []
        
        for i in range(MC_iter):
            
            data0,data1 = self.__data_generator.sample(self.__sample_size)
            


            if "dp" in self.__get_bound_types():
                dp_l, dp_u = calc_bounds_dp(data0, data1, handle_errors = self.get_handle_errors_dp())
                lower_bounds_dp.append(dp_l)
                upper_bounds_dp.append(dp_u)

            if "Bhattacharyya" in self.__get_bound_types():
                l, u = calc_Bhattacharyya_bounds(data0, data1, handle_errors = self.get_handle_errors_Bha())
                lower_bounds_Bha.append(l)
                upper_bounds_Bha.append(u)
            

            if "tight" in self.__get_bound_types() or "Bhatt_knn" in self.__get_bound_types():
                p0, p1= calc_knn_densities(data0, data1, self.__knn_num)
                prior0 = len(data0)/ (len(data0) + len(data1))
                prior1 = len(data1) / (len(data0) + len(data1))

                if "tight" in self.__get_bound_types():
                    l, u = calc_tight_bounds_knn(p0, p1, prior0, prior1, alpha=self.__tight_bounds_alpha)
                    lower_bounds_tight.append(l)
                    upper_bounds_tight.append(u)

                if "Bhatt_knn" in self.__get_bound_types():
                    a, b = calc_Bhatt_knn_bounds(p0, p1, prior0, prior1,  handle_errors= self.get_handle_errors_Bha_knn())
                    lower_bounds_Bha_knn.append(a)
                    upper_bounds_Bha_knn.append(b)
                


            if "Mahalanobis" in  self.__get_bound_types():
                a =calc_Mahalanobis_upper(data0, data1)
                upper_bounds_Maha.append(a)

            if "influence" in self.__get_bound_types():
                estim = eng.hellingerDivergence(data0, data1,[], [],  nargout= 1)
                BC = 1 - estim
                up = 1/2 * BC 
                low = 1/2 - 1/2 * np.sqrt(1- BC**2)

                lower_bounds_inf.append(low)
                upper_bounds_inf.append(up)
            
            if "enDive" in self.__get_bound_types():
                Dp = eng.EnDive(data0, data1, 'type', "DP",'quiet', 'kernel', self.__kernel,'est', 2, nargout= 1)
                if Dp > 1: ### this is kind of ghetto to do 
                    Dp = 1
                    
                if Dp > 0:
                    u = 1/2 -1/2 *Dp
                    l = 1/2 - 1/2 * np.sqrt(Dp)
                else:
                    u, l = 1/2 ,1/2
                lower_bounds_enDive.append(l)
                upper_bounds_enDive.append(u)

        eng.quit()
        return lower_bounds_dp, upper_bounds_dp, lower_bounds_tight, upper_bounds_tight, lower_bounds_Bha, upper_bounds_Bha, lower_bounds_Bha_knn, upper_bounds_Bha_knn, upper_bounds_Maha, lower_bounds_inf, upper_bounds_inf, lower_bounds_enDive, upper_bounds_enDive


    def __parallel_simulation(self, MC_iter, num_threads):
        MC_iter_per_thread = MC_iter // num_threads

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.__simulate_for_parallel, MC_iter_per_thread) for _ in range(num_threads)]

            for future in concurrent.futures.as_completed(futures):
                # print(future.result())
                lower_bounds_dp, upper_bounds_dp, lower_bounds_tight, upper_bounds_tight, lower_bounds_Bha, upper_bounds_Bha, lower_bounds_Bha_knn, upper_bounds_Bha_knn, upper_bounds_Maha,  lower_bounds_inf, upper_bounds_inf, lower_bounds_enDive, upper_bounds_enDive    = future.result()
                self.__lower_bounds_dp.extend(lower_bounds_dp)
                self.__upper_bounds_dp.extend(upper_bounds_dp)
                self.__lower_bounds_tight.extend(lower_bounds_tight)
                self.__upper_bounds_tight.extend(upper_bounds_tight)
                self.__lower_bounds_Bha.extend(lower_bounds_Bha)
                self.__upper_bounds_Bha.extend(upper_bounds_Bha)
                self.__lower_bounds_Bha_knn.extend(lower_bounds_Bha_knn)
                self.__upper_bounds_Bha_knn.extend(upper_bounds_Bha_knn)
                self.__upper_bounds_Maha.extend(upper_bounds_Maha)
                self.__lower_bounds_inf.extend(lower_bounds_inf)
                self.__upper_bounds_inf.extend(upper_bounds_inf)
                self.__lower_bounds_enDive.extend(lower_bounds_enDive)
                self.__upper_bounds_enDive.extend(upper_bounds_enDive)
