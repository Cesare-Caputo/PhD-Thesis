from stochasticdp import StochasticDP
import numpy as np
import pandas as pd
import copy
from garage_DP_helper import *
# from garage_demand import demand_stochastic_less, demand_stochastic, demand_stochastic_series, demand_static
from garage_cost import Exp_cost, opex
import math
import numpy as np
import pandas as pd
from garage_demand import *
from matplotlib import pyplot as plt
from scipy.optimize import *
# Parameters
T=20 #years
cc = 16000# Construction cost per parking space
cl = 3600000# Annual leasing land cost
#p = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth. The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
cr = 2000# Operating cost per parking space
#ct = []# Total construction cost
gc = 0.10# Growth in construction cost per floor above two floors
n0 = 200# Initial number of parking space per floor
p = 10000# Price per parking space
r = 0.12# Discount rate
fmin = 2# Minimum number of floors built
fmax = 8# Maximum number of floors built

D_1 = 750 # Projected year 1 demand
D_10 = 750 #additional demand by year 10
D_F= 250 # additional demand after year 10
T = 20 # project duration in years
alpha = D_10 + D_F; # Parameter for demand model showing difference between initial and final demand values
beta = -math.log(D_F/alpha)/(T/2 - 1) # Parameter for demand model showing growth speed of demand curve



DV_D = np.array([1, 1, 1, 0, 2, 2, 4])

# Design variables
#flex = []# Use the flexible design: 1 = "yes" 0 = "no"
#a1_4 = []# Expansion allowed in years 1 to 4: 1 = "yes" 0 = "no"
#a9_12 = []# Expansion allowed in years 9 to 12: 1 = "yes" 0 = "no"
#17_20 = []# Expansion allowed in years 17 to 20: 1 = "yes" 0 = "no"
#dr = []# Expansion rule: previous years with demand > capacity before expansion
#ft = []# Expansion rule: number of floors expanded by at year t
#f0 = []# Number of initial floors at year 0
#k = []
#E_cost = []

params = (20,1600,360000,2000, .1, 200, 10000, .12, 2, 8)

n_scenarios = 100

def cc_initial(f0):
    if f0 > 2:
        cc_start = cc * n0 * ((((1+gc)**(f0-1) - (1+gc))/gc)) + (2*n0*cc)
    else : 
        cc_start= f0*n0*cc
    return cc_start

def demand_series(T):
    years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
    demand_projections = pd.Series(index=years)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    for i in range(0,T+1): #initializing all ks to initial capacity
        demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s, rDfs)
    return demand_projections
    


def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = demand_stochastic_series(T)
    return scenario_df    

scenario_df = scenario_generator(n_scenarios)


################### Option A ################
import numpy as np
import pandas as pd
import copy
from garage_helper_functions import *
from garage_demand_functions import *
from garage_cost_functions import *
import math
from matplotlib import pyplot as plt
from scipy.optimize import *
from geneticalgorithm import geneticalgorithm as ga



# Parameters for cost model
T=20 #years
cc = 16000# Construction cost per parking space
cl = 3600000# Annual leasing land cost
cr = 2000# Operating cost per parking space
gc = 0.10# Growth in construction cost per floor above two floors
n0 = 200# Initial number of parking space per floor
p = 10000# Price per parking space
r = 0.12# Discount rate
fmin = 2# Minimum number of floors built
fmax = 8# Maximum number of floors built


#Parameters for demand model
D_1 = 750 # Projected year 1 demand
D_10 = 750 #additional demand by year 10
D_F= 250 # additional demand after year 10
T = 20 # project duration in years
alpha = D_10 + D_F; # Parameter for demand model showing difference between initial and final demand values
beta = -math.log(D_F/alpha)/(T/2 - 1) # Parameter for demand model showing growth speed of demand curve


# Design variables
#flex = []# Use the flexible design: 1 = "yes" 0 = "no"
#a1_4 = []# Expansion allowed in years 1 to 4: 1 = "yes" 0 = "no"
#a9_12 = []# Expansion allowed in years 9 to 12: 1 = "yes" 0 = "no"
#17_20 = []# Expansion allowed in years 17 to 20: 1 = "yes" 0 = "no"
#dr = []# Expansion rule: previous years with demand > capacity before expansion
#ft = []# Expansion rule: number of floors expanded by at year t
#f0 = []# Number of initial floors at year 0


# example demand vectors for debugging
DV_D = np.array([1, 1, 1, 0, 2, 2, 4])
DV_C = np.array([1, 1, 1, 0, 1, 1, 3])



# define the number of scenarios HERE that problem is optimized on
n_scenarios = 100 



def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = demand_stochastic_series(T)
    return scenario_df  
  
# generate explicitely here so that the same scenarios are passed for optimization objective function below
scenario_df = scenario_generator(n_scenarios)


def ENPV_garage_DR_scenarios(a): # this calculated ENPV using set of scenarios generated above for decision rule design
    ENPV_res =[]
    for i in range(n_scenarios):
        demand_scenario = scenario_df[i]
        ENPV2  = NPV_garage_GA_DR_scenarios(a, demand_scenario )
        ENPV_res.append(ENPV2)
    ENPV = np.mean(ENPV_res)   
    return -ENPV 
    
def ENPV_garage_fixed_scenarios(f0): # this calculated ENPV using set of scenarios generated above for fixed design
    ENPV_res =[]
    for i in range(n_scenarios):
        demand_scenario = scenario_df[i]
        ENPV2  = NPV_garage_inflex_scenarios(f0, demand_scenario )
        ENPV_res.append(ENPV2)
    ENPV = np.mean(ENPV_res)   
    return -ENPV 

#defining parameters for GA optimization, these should  be changed for different problems
algorithm_param_dr = {'max_num_iteration': 100,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.3,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':4}
    
 #setting up optimization of decision rules implementation 
varbound_dr = np.array([[1,1],[0,1],[0,1], [0,1] , [0,3], [0,3] , [0,4]]) # variable boundaries for GA implementation 
model_dr=ga(function=ENPV_garage_DR_scenarios, dimension=7, variable_type='int', algorithm_parameters=algorithm_param_dr,
            variable_boundaries=varbound_dr)  # setting up GA model instance
model_dr.run() # solution with GA

Ga_opt_dv = model_dr.best_variable # extracting optimal decision rule vector from optimization above

#setting up optimization of inflexible case
varbound_inflex = np.array([[0,9]])
model_inflex=ga(function=ENPV_garage_fixed_scenarios, dimension=1, variable_type='int',variable_boundaries=varbound_inflex, algorithm_parameters=algorithm_param_dr)  # setting up GA model instance for inflexible
model_inflex.run() # Inflexible solution with GA

f0_opt = model_inflex.best_variable


# Scipy DE implementation for benchmarking/ reliability of results
# can take really long to converge depending on number of scenarios used so can skip other than for benchmakring
bounds = [(1,1),(0,1), (0, 1), (0, 0), (0, 3), (0, 3), (0, 4)] # boundaries for scipy DE implementation 
mutation_cst = (0.1,1.99) # dithering implemented to speed up convergence
result_de_scipy = differential_evolution(func = ENPV_garage_DR_scenarios, bounds = bounds , callback=MinimizeStopper(), init = 'random', maxiter=1000,  mutation = mutation_cst ,  disp="True" )

DE_opt_dv = np.rint([ (result_de_scipy.x[0]) , (result_de_scipy.x[1]) , (result_de_scipy.x[2]) , (result_de_scipy.x[3]) , (result_de_scipy.x[4]) , (result_de_scipy.x[5]), (result_de_scipy.x[6])])



#calculating expected NPV for various solutions
ENPV_GA_opt =  ENPV_garage_DR_scenarios(Ga_opt_dv)
ENPV_DE_opt = NPV_garage_GA_DR(DE_opt_dv)
ENPV_inflex_opt = NPV_garage_GA_DR(DE_opt_dv)
print("ENPV for optimal policy with GA DR is Million $", ENPV_GA_opt*(10**-6))
print("ENPV for optimal policy with DE SciPy DR is Million $", ENPV_DE_opt*(10**-6))
print("ENPV for optimal policy with fixed floors is Million $", ENPV_inflex_opt*(10**-6))



# produce CDF comparing performance of flexiblity decision rules to inflexible
CDF_DR_inflex_scenarios(n_scenarios, scenario_df, Ga_opt_dv, f0_opt)

# Look at out of sample performance 
n_sim_outofsample = 2000
CDF_DR_inflex_outofsample(n_sim_outofsample, Ga_opt_dv, f0_opt)

















 ###################### Option b #################
    
a=NPV_garage_GA_DR(DV_D) 
bounds = [(1,1),(1,1), (0, 1), (0, 0), (0, 3), (0, 3), (0, 4)]
result = differential_evolution(func = NPV_garage_GA_DR_scenarios, bounds = bounds , maxiter=100)

Ga_opt_dv = np.array([ (result.x[0]) , (result.x[1]) , (result.x[2]) , (result.x[3]) , (result.x[4]) , (result.x[5]), (result.x[6])])
NPV_GA_opt, model_ga_opt = NPV_garage_GA_DR(Ga_opt_dv)
NPV_GA_opt_dv = NPV_garage_GA_DR(DV_D)
print("NPV for optimal policy with GA_DR Million $", NPV_GA_opt*(10**-6))
print("NPV for optimal prefound policy with GA_DR Million $", NPV_GA_opt_dv*(10**-6))



