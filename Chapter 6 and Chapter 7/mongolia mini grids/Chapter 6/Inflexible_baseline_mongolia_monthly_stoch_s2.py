# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:33:19 2021

@author: cesa_
"""

from electricity_demand import *
from minigrid_cost_rl_s2 import *
from electricity_generation import *
from heating_demand import *
from herder_migration import *
from heating_generation import *
import pandas as pd
import numpy as np
from mongolia_plotting import *
import scipy.optimize

# Parameters
T=30 #years
Tm = 360 # months in 20 years
r_yr = 0.06# Discount rate
r_month = ((1+r_yr)**(1/12)) - 1 
# static_capacity_pv = 363 #W of installef flat plate pv as starting capacity
# static_capacity_battery = 2000 #Wh 
# static_capacitY_inverter = 2630 # W
CF_avg = .20 # average capacity factor for mongolia solar from adb report
CF_dev = .01 # standard deviation of CF, assumed here but can be confirmed later


#defining indexes
years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241

#THIS IS THE ONLY THING TO CHANGE, THEN CAN RUN CODE FROM HERE TO GENERATE CDF
n_scenarios = 10
n_gers = 10



# parameters for electrcity generation predictions
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]




# heating parameters

# coal stove parameters
coal_HV_mj = 14.7 # MJ per kg
coal_HV_kj = 14.7 *1000 # kJ per kg
eff_trad_stove = .25
eff_improved_stove_h = .77 
eff_improved_stove_l = .54
kw_2_kj_perh = 3600 
coal_c02_emission_factor = 1.37 # tonnes of co2 per metric tonne coal burned
# these next values are in kj per kw hr to determine coal requirements
heat_input_trad = kw_2_kj_perh / eff_trad_stove
heat_input_improved_h = kw_2_kj_perh / eff_improved_stove_h
heat_input_improved_l = kw_2_kj_perh / eff_improved_stove_l




#parameters for financial evaluation
coal_price = 40 # $/tonne
coal_price_per_kg = coal_price*.001 # $/tonne
coal_trad_stove_capex = 13 # $
coal_improved_stove_capex_l = 129.5 
coal_improved_stove_capex_m = 151
coal_improved_stove_capex_h = 181


# ELECTRICAL HEATER TECHNICAL SPECS
eff_electric_stove = 1
avg_co2_emission_factor_grid = .7111 # tonnes Co2 per year
eh_lifetime = 13 # years USE LATER TO LOOK AT REPLACEMENT COSTS
eh_capex_kw = 152 # $ per kW
eh_capex_w = 152000 # $ per kW



n_lattice_walls = 5
area_ger_5_walls = 29.3 # m2
area_ger_8_walls = 72.6 # m2
yearly_demand_standard_ger_perm2 = 393 #/ kWh / m2/ yr
yearly_demand_improved_ger_perm2 = 206 #/ kWh / m2/ yr
heating_months = 8 # September 1 to May 1 each year

# generate deterministic demand and migration cluster radius profile and heating , demand values are foe 10 gers
demand_projections = demand_static_series_months_ngers(Tm, n_gers)
cluster_radius_projections = migration_cluster_radius_static_series(Tm)



# heating demand kept determinstic as no reason to suggest otherwise
heating_demand_projections =  monthly_heat_demand_fromtemp_ngers(month_temp , yearly_demand_standard_ger_perm2, area_ger_5_walls, n_gers)


def inflex_mongolia_df_stoch(initial_design):
    n_gers = 10
    starting_pv_capacity = initial_design[0] * n_gers
    starting_eh_capacity = initial_design[1] * n_gers
    money_weighing_factor = 1
    emission_weighing_factor = 0
    coal_c02_emission_factor = 1.37 # tonnes of co2 per metric tonne coal burned
    cost_df = pd.DataFrame(index = months, columns = [ 'Total' , 'Capex' ,'Mismatch' ,'EH(kWh)' , 'Grid($)' , 'Grid(kWh)', 'Opex' ,'Coal' , 'Coal Cost', 'CO2', 'LS %'])
    #eh_heat = 0 # inflexible , no electric heaters available
    cost = 0 
    demand_series = electricity_monthly_demand_stochastic_less_series_ngers(Tm, n_gers)
    heating_demand_series = heating_demand_projections
    cf_pv_month = CF_pv_monthly
    coal_price = coal_price_per_kg
    static_capacity_pv = starting_pv_capacity
    static_capacity_battery = battery_per_pv_inflex(static_capacity_pv)
    static_capacity_inverter , inverter_capex = expansion_impact_inverter(static_capacity_pv)
    #these times are in months, corresponding to expected replacement years
    batt_repl_yr = list(range(48, 360, 48))
    pv_repl_yr = [242]
    inverter_repl_yr =[180]
    eh_repl_yr = [181]
    for i in range(Tm + 1):
        month_index = monthly_index_fromtimestep(i)
        pv_cf = randomised_cf_norm_month(month_index, CF_pv_monthly,CF_pv_monthly_dev) # stochastic capacity factor
        electricity_demand = demand_series[i]
        heat_demand = heating_demand_series[month_index]
        static_capacity_pv_ngers = static_capacity_pv * n_gers
        electricity_generated_monthly_kwh = normalised_monthly_stoch_electricity_generation_kwh (pv_cf, static_capacity_pv)
        extra_electricity = np.max([(electricity_generated_monthly_kwh - electricity_demand) , 0])
        #NOTE THAT LOAD SHEDDING POTENTIAL EVENTS ARE INCLUDED IN THIS
        grid_eh_purchase, eh_heat_grid , eh_heat_res , simulated_load_shed = grid_electricity_interaction_inflex_2(extra_electricity, starting_eh_capacity, heat_demand , i)
        eh_heat_total = eh_heat_res + eh_heat_grid
        #coal_cost = monthly_coal_expenditure(heat_demand, eh_heat, coal_HV_kj,  eff_trad_stove, coal_price)
        coal_mass = monthly_coal_requirement(heat_demand, eh_heat_total, coal_HV_kj,  eff_trad_stove)
        coal_cost = coal_mass * coal_price_per_kg
        mismatch = mismatch_penalty(electricity_generated_monthly_kwh, demand_series[i])
        cost_df.loc[i, 'LS %' ] = simulated_load_shed *100
        cost_df.loc[i, 'Grid($)'] = grid_eh_purchase
        cost_df.loc[i, 'Mismatch'] = mismatch
        cost_df.loc[i, 'Opex'] = monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) 
        cost_df.loc[i, 'Coal Cost'] = coal_cost
        cost_df.loc[i, 'Coal']= coal_mass
        cost_df.loc[i, 'CO2'] = carbon_footprint_inflex(coal_mass,eh_heat_grid) 
        cost_df.loc[i, 'EH(kWh)']= eh_heat_total
        cost_df.loc[i, 'Grid(kWh)'] = eh_heat_grid
        if i == Tm + 1:
            pv_salvage_value = salvage_pv(static_capacity_pv, 10)
            cost_df.loc[i, 'Total'] = (mismatch - pv_salvage_value + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter) + coal_cost)/ ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = - pv_salvage_value
        elif i ==0 :
            cost_df.loc[i, 'Total'] = pv_capex_inflex(starting_pv_capacity) + eh_capex_inflex(starting_eh_capacity) + battery_capex_inflex(static_capacity_battery) + inverter_capex_inflex(static_capacity_inverter)
            cost_df.loc[i, 'Capex'] = pv_capex_inflex(starting_pv_capacity) + eh_capex_inflex(starting_eh_capacity) + battery_capex_inflex(static_capacity_battery) + inverter_capex_inflex(static_capacity_inverter)
        elif i in batt_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + battery_capex_inflex(static_capacity_battery))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = battery_capex_inflex(static_capacity_battery)
        elif i in pv_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + pv_capex_inflex(starting_pv_capacity))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = pv_capex_inflex(static_capacity_pv)
        elif i in inverter_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + inverter_capex_inflex(static_capacity_inverter))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = inverter_capex_inflex(static_capacity_inverter)
        elif i in eh_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + eh_capex_inflex(starting_eh_capacity))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = eh_capex_inflex(starting_eh_capacity)      
        else:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost)
        cost_df = cost_df.fillna(0)
    return cost_df



def inflex_mongolia_capacity_opt_stoch(initial_design):
    cost_df = inflex_mongolia_df_stoch(initial_design)
    lcc = cost_df["Total"].sum()
    return lcc





def inflex_mongolia_c02_opt_stoch(initial_design):
    cost_df = inflex_mongolia_df_stoch(initial_design)
    lcc = cost_df["CO2"].sum()
    return lcc



def elcc_inflexible(initial_design, n_scenarios):
    elccs =[]
    for i in range(n_scenarios):
        lcc = inflex_mongolia_df_mo(initial_design)
        elccs.append(lcc)
    elcc = np.mean(elccs)
    return elcc




def elcc_inflex_opt(initial_design):
    n_scenarios = 10
    elccs =[]
    for i in range(n_scenarios):
        lcc = inflex_mongolia_capacity_opt_stoch(initial_design)
        elccs.append(lcc)
    elcc = np.mean(elccs)
    return elcc

def elc02_inflex_opt(initial_design):
    n_scenarios = 10
    elccs =[]
    for i in range(n_scenarios):
        lcc = inflex_mongolia_c02_opt_stoch(initial_design)
        elccs.append(lcc)
    elcc = np.mean(elccs)
    return elcc




def elcc_inflex_pvopt(pv_capacity):
    n_scenarios = 10
    elccs =[]
    initial_design = np.array([pv_capacity, 0])
    for i in range(n_scenarios):
        lcc = inflex_mongolia_df_mo(initial_design)
        elccs.append(lcc)
    elcc = np.mean(elccs)
    return elcc

# x0 = np.array([380, 0])

# a = inflex_mongolia_df_stoch(x0)


# inflex_mongolia_c02_opt_stoch(x0)



# pv0_homer = 363
# pv0_ga = 8850
# eh_0_test = 1000

# x0 = np.array([pv0_ga, 1000])
# x1 = np.array([0, 4000])

# x0 = np.array([380, 0])
# x1 = np.array([863, 5000])
# x3 = np.array([86300, 50000])

# xweird = np.array([4979, 280])

# b=inflex_mongolia_df_stoch(xweird)

# # # x_mo = np.array([0, 10000])

# # a = inflex_mongolia_df(x1)
# # print(inflex_mongolia_capacity_opt(x1))


# # inflex_df_mo_0 = inflex_mongolia_df_mo(x0)
# # # inflex_df_mo_1 = inflex_mongolia_df_mo(x1)
# # print(inflex_mongolia_capacity_opt(x0))



# # print(inflex_df_mo_0)
# # print(inflex_df_mo_1)


# print(elc02_inflex_opt(x3))

# print(inflex_mongolia_c02_opt_stoch(x3))

# print(inflex_mongolia_capacity_opt(x0))
# print(inflex_mongolia_c02_opt(x0))





from scipy.optimize import differential_evolution, minimize
results = dict()
bounds = [(200,1000) , (0,700)]
# pv_bounds =  [(200,1000)]
# bnds = ((200,3000))
results['GA'] = differential_evolution(elcc_inflex_opt, bounds)


 


# # results['NM']=  minimize(inflex_mongolia_df_mo, x0, method='Nelder-Mead',  
# #                             options={'disp':True,'maxiter':1001})


# # print(results['NM'])
print(results['GA'])


cost_df = inflex_mongolia_df_stoch(results['GA'].x)

#%% plotting CF evolution
# 'Total' , 'Capex' ,'Mismatch' ,'EH(kWh)' , 'Grid($)' , 'Grid(kWh)', 'Opex' ,'Coal' , 'Coal Cost', 'CO2', 'LS %'
# n_yr = 12
# cost_df_yearly = cost_df.groupby(cost_df.index //n_yr).sum()
# full_cost_df = cost_df_yearly

# co2_evolution = full_cost_df['CO2']

# full_cost_df.drop('Total' , inplace = True, axis = 1)
# full_cost_df.drop('LS %' , inplace = True, axis = 1)
# full_cost_df.drop('EH(kWh)' , inplace = True, axis = 1)
# full_cost_df.drop('Coal' , inplace = True, axis = 1)
# full_cost_df.drop('Grid(kWh)' , inplace = True, axis = 1)
# full_cost_df.drop('CO2' , inplace = True, axis = 1)
# full_cost_df["Opex"] = full_cost_df["Opex"]*200


# #correct final year values for plot
# full_cost_df.iloc[30] = full_cost_df.iloc[29] *.95
# co2_evolution.iloc[30] = co2_evolution.iloc[29] *.95

# # #HERE DEFINE YEAR 0 COSTS TO FILL PANDAS DATAFRAME AND MAKE MORE SENSE FROM GRAPH
# # yr0_row_inflex = pd.DataFrame({'Total' : [0] , 'Capex' : [24300] , 'Mismatch' : [0], 'Opex' : [0] , 'Coal' : [0] })
# # #yr0_row_inflex2 = pd.DataFrame({'Total' : [0] , 'Capex' : [2430] , 'Mismatch' : [0], 'Opex' : [0] , 'Coal' : [0] })
# # full_cost_df = pd.concat([yr0_row_inflex, cost_df_yearly]).reset_index(drop = True)



# # full_cost_df.drop('Total' , inplace = True, axis = 1)
# # full_cost_df.drop(full_cost_df.index[21] , inplace = True, axis = 0)


# #full_cost_df["Coal"] = full_cost_df["Coal"]/1.1575

# # full_cost_df["Capex"] = full_cost_df["Capex"]*2

# a = full_cost_df.plot.bar(stacked = True)
# a.legend( loc = 'lower center' , ncol = 5 ,bbox_to_anchor = (.5 , -.4))
# a.set_xlabel('Years')
# a.set_ylabel('Cost ($)')
# a.set_title('Scenario 1: Cumulative Cost Breakdown- Stochastic Baseline - Full System')

# ax2=a.twinx()
# ax2.plot(co2_evolution, color = "red")
# ax2.set_ylabel('Yearly Tonnes CO2 emitted')
# plt.show()

# # save the plot as a file
# fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
#             format='jpeg',
#             dpi=100,
#             bbox_inches='tight')



# full_cost_df["Capex"][10] = 1205
# full_cost_df["Capex"][15] = 13510


#inflex_monthly_cf_plot(demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)
#inflex_yearly_cf_bar_det(demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)


# # x_old = np.array([363, 300])


#%%# # # # # test outputs

# # print(inflex_mongolia_df_mo(results['NM'].x))
# print(inflex_mongolia_df_mo(results['GA'].x))
# # print(inflex_mongolia_df_mo(x_old))


# xt = np.array([471, 300])
# xga = np.array([525, 44])
# xga0 = np.array([525, 0])
# xga_pv = np.array([505, 0])
# xga3 = np.array([609, 0])


# print(elcc_inflex_opt(results['GA'].x))
# print(elcc_inflex_opt(xt))
# print(elcc_inflex_opt(xga0))
# print(elcc_inflex_opt(xga))
# print(elcc_inflex_opt(xga_pv))
# print(elcc_inflex_opt(xga3))
# # a = inflex_mongolia_df(results['NM'].x)
# # b = inflex_mongolia_df(results['GA'].x)
# # c = inflex_mongolia_df(x_old)




# print(inflex_mongolia_df_stoch(xga))
# print(inflex_mongolia_df_stoch(xt))





#%% MULTI OBJECTIVE OPTIMIZATION OF INFLEXIBLE BASELINE

# from pymoo.algorithms.nsga2 import NSGA2
# from pymoo.model.problem import Problem
# from pymoo.optimize import minimize
# from pymoo.visualization.scatter import Scatter
# from pymoo.util.misc import stack
# from pymoo.util.termination.default import MultiObjectiveDefaultTermination ,MultiObjectiveSpaceToleranceTermination
# from pymoo.model.callback import Callback
# termination = MultiObjectiveDefaultTermination(
#     x_tol=100,
#     cv_tol=100,
#     f_tol=10000,
#     nth_gen=5,
#     n_last=100,
#     n_max_gen=100,
#     n_max_evals=100000
# )

# # termination = MultiObjectiveSpaceToleranceTermination(tol=1000,
# #                                                       n_last=30,
# #                                                       nth_gen=5,
# #                                                       n_max_gen=5,
# #                                                       n_max_evals=5)

# class MyCallback(Callback):

#     def __init__(self) -> None:
#         super().__init__()
#         self.n_evals = []
#         self.opt = []

#     def notify(self, algorithm):
#         self.n_evals.append(algorithm.evaluator.n_eval)
#         self.opt.append(algorithm.opt[0].F)
        
# norm_range_cost = 350000
# norm_range_co2 = 6000

# class MyProblem(Problem):
#     def __init__(self):
#         super().__init__(n_var=2,
#                           n_obj=2,
#                           n_constr=0,
#                           xl=np.array([10, 20]),
#                           xu=np.array([5000, 10000]),
#                           elementwise_evaluation=True)
#     def _evaluate(self, x, out, *args, **kwargs):
#         f1 = inflex_mongolia_capacity_opt_stoch(x) / norm_range_cost
#         f2 = inflex_mongolia_c02_opt_stoch(x) /norm_range_co2

#         out["F"] = [f1, f2]


# problem = MyProblem()
# callback = MyCallback()
# algorithm = NSGA2(pop_size=20)

# res = minimize(problem,
#                 algorithm,
#                 callback=callback,
#                 termination = termination,
#                 verbose=True,
#                 seed=1,
#                 save_history=True)


# # print(minimize.indicator)


# # get the pareto-set and pareto-front for plotting
# ps = problem.pareto_set(use_cache=False, flatten=False)
# pf = problem.pareto_front(use_cache=False, flatten=False)


# # Objective Space
# plot = Scatter(title = "Stochastically Optimal Inflexible Design: Pareto Frontier")
# plot.add(res.F).show()
# if pf is not None:
#     plot.add(pf, plot_type="line", color="black", alpha=0.7)
# plot.
# plot.show()


# # #CONVERT BACK TO ORIGINAL UNITS TO UNDERSTANDS VALUES
# for i in range(len(res.F)):
#     res.F[i][0] = res.F[i][0] * norm_range_cost
#     res.F[i][1] = res.F[i][1] * norm_range_co2   
    
    
    
# # # res.F[0] =res.F[0]/ norm_range_cost

# for i in range(len(res.X)):
#     x = res.X[i] 
#     print("System cost for" , i , "pareto solution is" , inflex_mongolia_capacity_opt_stoch(x))
#     print("System CO2 for" , i , "pareto solution is" , inflex_mongolia_c02_opt_stoch(x))



# res.X[1]

# a =inflex_mongolia_df_stoch(res.X[1])

# #PLOT CONVERGENCE


# plt.title("Convergence")
# plt.plot(callback.n_evals, callback.opt, "--")
# plt.yscale("log")
# plt.show()


# n_evals = []    # corresponding number of function evaluations\
# F = []          # the objective space values in each generation
# cv = []         # constraint violation in each generation
# Xs = []

# # iterate over the deepcopies of algorithms
# for X in res.history:
#     xs.append(res.X)
#     # store the number of function evaluations
#     n_evals.append(algorithm.evaluator.n_eval)
    
#     # retrieve the optimum from the algorithm
#     opt = algorithm.opt

#     # store the least contraint violation in this generation
#     cv.append(opt.get("CV").min())

#     # filter out only the feasible and append
#     feas = np.where(opt.get("feasible"))[0]
#     _F = opt.get("F")[feas]
#     F.append(_F)





# print("Baseline CO2 emissions are in tonnes " , inflex_mongolia_co2_opt(pv0_ga))

#inflex_df = inflex_mongolia_df(x0)

# total_coal = inflex_df["Coal"].sum()
# total_c02 = inflex_df["CO2"].sum()
# print("Total Carbon used in kg is " , total_coal)
# print("Resulting in tonnes CO2 emitted of " , total_c02)

# from pymoo.util.running_metric import RunningMetric


# running = RunningMetric(delta_gen=10,
#                         n_plots=4,
#                         only_if_n_plots=True,
#                         key_press=False,
#                         do_show=True)

# for algorithm in res.history:
#     running.notify(algorithm)




#OTHER OLD OPTIMIZATION CODE FROM GAGARAGE EXMAPLE

#results['SLSQP']=  minimize(NPV_garage, plan_bh ,method='SLSQP', bounds = bnds, options={'disp':True,'maxiter':1001,'ftol':1E-5})   
# #results['shgo'] = optimize.shgo(NPV_garage, bnds)
# #results['DA'] = dual_annealing(NPV_garage, bounds=bnds, seed=1234)
# #results['Dual annealing '] = np.rint(results['DA'].x)
# results_GA = differential_evolution(ENPV_MC, bnds)
# results['GA round2'] = np.rint(results_GA.x)
# #results_GA_stoc = differential_evolution(ENPV_MC, bnds)
# #results['GA round stoc'] = np.rint(results_GA_stoc.x)

# #results_GA_BH = basinhopping(ENPV_MC, plan_bh, niter =100)
                                         
# #resbf =  optimize.brute(NPV_obj_array, ranges, finish =None)
# #results['Brute force'] = resbf
# #print(res3)
# #print ("Optimized DV is :" , result.x )
# #print( "leading to NPV of: " , result.fun)
# #print(result)

# NPV_GA_opt = NPV_garage(results['GA round2'])
# # print("leading to NPV of: " , NPV_GA_opt)
# print("leading to ENPV of: " , ENPV_MC(results['GA round2']))





# a =  inflex_mongolia_opt(500)





# cost_df1 = total_system_cost_determinstic_monthly_inflex (demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)

# cost_df2 = total_system_cost_determinstic_monthly_inflex_disc (demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)

# n_yr = 12
# cost_df_yearly = cost_df1.groupby(cost_df1.index //n_yr).sum()

# #HERE DEFINE YEAR 0 COSTS TO FILL PANDAS DATAFRAME AND MAKE MORE SENSE FROM GRAPH
# yr0_row_inflex = pd.DataFrame({'Total' : [0] , 'Capex' : [24300] , 'Mismatch' : [0], 'Opex' : [0] , 'Coal' : [0] })
# #yr0_row_inflex2 = pd.DataFrame({'Total' : [0] , 'Capex' : [2430] , 'Mismatch' : [0], 'Opex' : [0] , 'Coal' : [0] })
# full_cost_df = pd.concat([yr0_row_inflex, cost_df_yearly]).reset_index(drop = True)



# full_cost_df.drop('Total' , inplace = True, axis = 1)
# full_cost_df.drop(full_cost_df.index[21] , inplace = True, axis = 0)


#full_cost_df["Coal"] = full_cost_df["Coal"]/1.1575

# full_cost_df["Capex"] = full_cost_df["Capex"]*2

# a = full_cost_df.plot.bar(stacked = True)
# a.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.5 , -.4))
# a.set_xlabel('Years')
# a.set_ylabel('Cost ($)')
# a.set_title('Scenario 1: Cumulative Cost Breakdown- Determinstic Baseline - Full System')


# full_cost_df["Capex"][10] = 1205
# full_cost_df["Capex"][15] = 13510


#inflex_monthly_cf_plot(demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)
#inflex_yearly_cf_bar_det(demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)
