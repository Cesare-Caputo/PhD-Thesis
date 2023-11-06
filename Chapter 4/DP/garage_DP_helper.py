# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:23:27 2020

@author: cesa_
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from garage_DP_class import Garage_Complete, cc_start
from backward_induction_dp import StochasticDP
from matplotlib import pyplot as plt
from garage_cost import Exp_cost
from garage_demand import demand_static,  demand_stochastic_less
import pandas as p
from Inflexible_baseline import NPV_garage_inflex, CF_model_out_inflex


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
fmax = 9# Maximum number of floors built
kmax = n0*fmax


def NPV_garage_DP_BI(policy , demand = ''):
    # calculates the NPV when following a given determinstic policy obtained viA BACKWARDS INDUCTION, so is a dictionary with sets containing policy
    NPV = 0
    years = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
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
    fmax = 9# Maximum number of floors built
    T = 20
    kmax = n0*fmax
    k = pd.Series(index=years, name = 'capacity', dtype ='float64')
    revenue = pd.Series(index=years, dtype ='float64')
    CF = pd.Series(index=years,name ='Cash Flow', dtype ='float64')
    demand_projections = pd.Series(index=years, name = 'demand', dtype ='float64')
    opex = pd.Series(index=years, dtype ='float64')
    fixed_costs = pd.Series(index=years, name = 'Fixed Costs',  dtype ='float64')
    E_cost = pd.Series(index=years, name = ' Construction costs', dtype ='float64')
    actions = pd.Series(index=years,name = 'action taken', dtype ='float64')
    E_cost[T]=0
    k[0] = 0
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10

    
    for i in range(0,21):
        if i ==20:
            k[i] = k[i-1]
            actions[i] = np.array(list(policy[i, k[i]]))
            E_cost[i] = Exp_cost(k[i], actions[i] ) #remember to adjust this to year 0 AFTER BUG FIXED
        else:
            capacity = k[i]
            actions[i] = np.array(list(policy[i, k[i]]))
            k[i+1] = k[i] + 200*actions[i] 
            E_cost[i] = Exp_cost(k[i], actions[i] )

    
    for i in range(1,21): #initializing all ks to initial capacity
        demand_projections[0] = 0
        if demand == 'stochastic':
            demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s,rDfs)
        else:
            demand_projections[i] = demand_static(i)        
    
   
    E_cost[0] = cc_start(actions[0])
    for i in range(0,T+1):
        if i ==T:
            fixed_costs[i] = 0 #no leasing paid in last year
        else : 
            fixed_costs[i] = cl # leasing paid all years including 0CF
    for i in range(0,T+1): #initializing all ks to initial capacity but maybe change here to let k[0] = 0 
        revenue[i] = np.minimum(demand_projections[i], k[i])*p
        opex[i] = k[i] * cr
        CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
        NPV += CF[i]/((1+r)**i)
        model = pd.concat([CF, k, actions, demand_projections,  E_cost ], axis = 1)
    return  NPV, model

def NPV_garage_DP_BI_constantpolicy(policy , demand_scenario):
    # calculates the NPV when following a given determinstic policy obtained viA BACKWARDS INDUCTION, so is a dictionary with sets containing policy
    NPV = 0
    years = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
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
    fmax = 9# Maximum number of floors built
    T = 20
    kmax = n0*fmax
    k = pd.Series(index=years, name = 'capacity', dtype ='float64')
    revenue = pd.Series(index=years, dtype ='float64')
    CF = pd.Series(index=years,name ='Cash Flow', dtype ='float64')
    demand_projections = pd.Series(index=years, name = 'demand', dtype ='float64')
    opex = pd.Series(index=years, dtype ='float64')
    fixed_costs = pd.Series(index=years, name = 'Fixed Costs',  dtype ='float64')
    E_cost = pd.Series(index=years, name = ' Construction costs', dtype ='float64')
    actions = pd.Series(index=years,name = 'action taken', dtype ='float64')
    E_cost[T]=0
    k[0] = 0
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    demand_projections = demand_scenario
    
    for i in range(0,21):
        if i ==20:
            k[i] = k[i-1]
            actions[i] = np.array(list(policy[i, k[i]]))
            E_cost[i] = Exp_cost(k[i], actions[i] ) #remember to adjust this to year 0 AFTER BUG FIXED
        else:
            capacity = k[i]
            actions[i] = np.array(list(policy[i, k[i]]))
            k[i+1] = k[i] + 200*actions[i] 
            E_cost[i] = Exp_cost(k[i], actions[i] )

    
   
    
   
    E_cost[0] = cc_start(actions[0])
    for i in range(0,T+1):
        if i ==T:
            fixed_costs[i] = 0 #no leasing paid in last year
        else : 
            fixed_costs[i] = cl # leasing paid all years including 0CF
    for i in range(0,T+1): #initializing all ks to initial capacity but maybe change here to let k[0] = 0 
        revenue[i] = np.minimum(demand_projections[i], k[i])*p
        opex[i] = k[i] * cr
        CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
        NPV += CF[i]/((1+r)**i)
        model = pd.concat([CF, k, actions, demand_projections,  E_cost ], axis = 1)
    return NPV, model


def NPV_garage_DP_BI_scenario(policy , demand_scenario):
    # calculates the NPV when following a given determinstic policy obtained viA BACKWARDS INDUCTION, so is a dictionary with sets containing policy
    NPV = 0
    years = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
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
    fmax = 9# Maximum number of floors built
    T = 20
    kmax = n0*fmax
    k = pd.Series(index=years, name = 'capacity', dtype ='float64')
    revenue = pd.Series(index=years, dtype ='float64')
    CF = pd.Series(index=years,name ='Cash Flow', dtype ='float64')
    demand_projections = pd.Series(index=years, name = 'demand', dtype ='float64')
    opex = pd.Series(index=years, dtype ='float64')
    fixed_costs = pd.Series(index=years, name = 'Fixed Costs',  dtype ='float64')
    E_cost = pd.Series(index=years, name = ' Construction costs', dtype ='float64')
    actions = pd.Series(index=years,name = 'action taken', dtype ='float64')
    E_cost[T]=0
    k[0] = 0
    demand_projections = demand_scenario

    
    for i in range(0,21):
        if i ==20:
            k[i] = k[i-1]
            actions[i] = np.array(list(policy[i, k[i]]))
            E_cost[i] = Exp_cost(k[i], actions[i] ) #remember to adjust this to year 0 AFTER BUG FIXED
        else:
            capacity = k[i]
            actions[i] = np.array(list(policy[i, k[i]]))
            k[i+1] = k[i] + 200*actions[i] 
            E_cost[i] = Exp_cost(k[i], actions[i] )

        
   
    E_cost[0] = cc_start(actions[0])
    for i in range(0,T+1):
        if i ==T:
            fixed_costs[i] = 0 #no leasing paid in last year
        else : 
            fixed_costs[i] = cl # leasing paid all years including 0CF
    for i in range(0,T+1): #initializing all ks to initial capacity but maybe change here to let k[0] = 0 
        revenue[i] = np.minimum(demand_projections[i], k[i])*p
        opex[i] = k[i] * cr
        CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
        NPV += CF[i]/((1+r)**i)
        model = pd.concat([CF, k, actions, demand_projections,  E_cost ], axis = 1)
    return  NPV, model



#showing capacity evolutions 
def capacity_evolution_plots(policy , DV, f0,  demand  = ''):
    NPV_bi, model_bi = NPV_garage_DP_BI(policy , demand = '')
    NPV_ga_dr, model_ga_dr = NPV_garage_GA_DR(DV, demand = '')
    model_inflex = CF_model_out_inflex(f0)
    NPV_inflex = NPV_garage_inflex(f0)
    plt.figure(figsize=(12,5))
    plt.xlabel('Years')
    plt.ylabel('Capacity/Demand')
    plt.title('Demand and Capacity evolution over project lifetime for various designs')
    ax1=model_bi.capacity.plot(color='blue', grid=True, label='DP Policy ')
    ax2=model_bi.demand.plot(color='red', grid=True, label='Static Demand')
    ax3 = model_ga_dr.capacity.plot(color ='orange', grid = True, label  ='GA DR')
    ax4 = model_inflex.Capacity.plot(color='magenta', grid=True, label='Fixed ')
    plt.text(15, 750, 'NPV DP Policy : {:.2f} Million $'.format(NPV_bi/1000000))
    plt.text(15, 500, 'NPV GA DR : {:.2f} Million $'.format(NPV_ga_dr/1000000))
    plt.text(15, 250, 'NPV Inflexible : {:.2f} Million $'.format(NPV_inflex/1000000))
    ax1.legend(loc=2)
    ax2.legend(loc=2)
    ax3.legend(loc=2)
    ax4.legend(loc=2)
    plt.show()
    
    
    
def ENPV_MC_CDF(policy, nsim): # input a policy and number simualtions used for CDF
    ENPV_res =[]
    for i in range(nsim):
        ENPV2, model2 = NPV_garage_DP_BI(policy, demand = 'stochastic')
        ENPV_res.append(ENPV2)  
    EPV = np.array(ENPV_res)
    ENPV = np.mean(EPV)
    fig, ax = plt.subplots(figsize=(8, 4))
    cdf = ax.hist(EPV, 100, density=True, histtype='step',
                       cumulative=True, label='DP policy performance')
    plt.axvline(ENPV, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(ENPV*1.05, max_ylim*0.9, 'ENPV: {:.2f} million $'.format(ENPV/1000000))
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_title('CDF of design solution')
    ax.set_xlabel('NPV of policy($)')
    ax.set_ylabel('Likelihood of occurrence')
    return cdf


def ENPV_MC(nsim, policy):
    ENPV_res =[]
    for i in range(nsim):
        ENPV2, model2 = NPV_garage_DP_BI(policy, demand = 'stochastic')
        ENPV_res.append(ENPV2)
    ENPV = np.mean(ENPV_res)   
    return ENPV


def CDF_DP_policy_fixed(nsim, policy_d , f0):
    NPVd =[]
    NPVf =[]
    for i in range(nsim):
        NPV_d , model_d = NPV_garage_DP_BI(policy_d, demand = 'stochastic') 
        NPV_f = NPV_garage_inflex(f0, demand = 'stochastic') 
        NPVd.append(NPV_d)
        NPVf.append(NPV_f)
    EPVd = np.array(NPVd)
    EPVf = np.array(NPVf)
    ENPVd = np.mean(EPVd)
    ENPVf = np.mean(EPVf)         

    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_d = bx.hist(EPVd, 100, density=True, histtype='step',
                        cumulative=True, label='Flexible DP policy')
    cdf_s = bx.hist(EPVf, 100, density=True, histtype='step',
                        cumulative=True, label='Inflexible Design')
    plt.axvline(ENPVd, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-10000000, max_ylim*0.75, 'ENPV DP Policy : {:.2f} Million $'.format(ENPVd/1000000))
    plt.axvline(ENPVf, color='darkorange', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-10000000, max_ylim*0.6, 'ENPV  Inflexible design : {:.2f} Million $'.format(ENPVf/1000000))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions : Dp policy comparison to inflexible')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_d


def NPV_garage_predefined_policy(policy , demand = ''):
    # calculates the NPV when following a given determinstic policy 
    # policy is specified at first as set of 
    NPV = 0
    years = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
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
    fmax = 9# Maximum number of floors built
    T = 20
    kmax = n0*fmax
    k = pd.Series(index=years, name = 'capacity', dtype ='float64')
    revenue = pd.Series(index=years, dtype ='float64')
    CF = pd.Series(index=years,name ='Cash Flow', dtype ='float64')
    demand_projections = pd.Series(index=years, name = 'demand', dtype ='float64')
    opex = pd.Series(index=years, dtype ='float64')
    fixed_costs = pd.Series(index=years, name = 'Fixed Costs',  dtype ='float64')
    E_cost = pd.Series(index=years, name = ' Construction costs', dtype ='float64')
    actions = pd.Series(index=years,name = 'action taken', dtype ='float64')
    E_cost[T]=0
    k[0] = 0
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10

    
    for i in range(0,21):
        if i ==20:
            k[i] = k[i-1]
            actions[i] = policy[i]
            E_cost[i] = Exp_cost(k[i], actions[i] ) #remember to adjust this to year 0 AFTER BUG FIXED
        else:
            capacity = k[i]
            actions[i] = policy[i]
            k[i+1] = k[i] + 200*actions[i] 
            E_cost[i] = Exp_cost(k[i], actions[i] )

    
    for i in range(1,21): #initializing all ks to initial capacity
        demand_projections[0] = 0
        if demand == 'stochastic':
            demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s,rDfs)
        else:
            demand_projections[i] = demand_static(i)        
    
   
    E_cost[0] = cc_start(actions[0])
    for i in range(0,T+1):
        if i ==T:
            fixed_costs[i] = 0 #no leasing paid in last year
        else : 
            fixed_costs[i] = cl # leasing paid all years including 0CF
    for i in range(0,T+1): #initializing all ks to initial capacity but maybe change here to let k[0] = 0 
        revenue[i] = np.minimum(demand_projections[i], k[i])*p
        opex[i] = k[i] * cr
        CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
        NPV += CF[i]/((1+r)**i)
        model = pd.concat([CF, k, actions, demand_projections,  E_cost ], axis = 1)
    return  NPV, model



def NPV_garage_GA_DR_scenarios(a, demand_scenario , demand = ''):
        a.reshape((-1,1))
        flex = round(a[0])
        a1_4= round(a[1])
        a9_12= round(a[2])
        a17_20= round(a[3])
        dr = round(a[4])
        ft= round(a[5])
        f0 = round(a[6])
        n0 = 200
        kmax = 1800
        NPV = 0

    
        cc_initial = cc_start(f0)        
        
        years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
        k = pd.Series(index=years , name = 'capacity')
        revenue = pd.Series(index=years)
        CF = pd.Series(index=years, name = ' Cash flow')
        demand_projections = demand_scenario
        opex = pd.Series(index=years)
        fixed_costs = pd.Series(index=years)
        E_cost = pd.Series(index=years, name = ' expansion cost')
        plan=pd.Series(index=years) #this shows whether expansion is allowed each year
        
        for i in range(0,T+1):
            plan[i] = 1
            if i ==0:
                k[i] = 0 #initialize capacity to 0
            else : 
                k[i] = f0*n0  # initialize all others to initial
            for i in range(1,5):
                plan[i] = a1_4
            for i in range(9,13):
                plan[i] = a9_12   
            for i in range(17,21):
                plan[i] = a17_20
                

        
        
        for i in range(1,T):
            if flex == 1 and (k[i]+ ft*n0) < kmax and plan[i] ==1:
                if dr == 1:
                    if demand_projections[i]> k[i] :
                        k[i + 1] = k[i] + ft*n0                                    #capacity added after one year
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                        k[i+1]=k[i]
                        E_cost[i] = 0
                elif dr == 2:
                    if demand_projections[i]> k[i] and demand_projections[i-1]> k[i-1] :
                        k[i + 1] = k[i] + ft*n0                                    #capacity added after two years
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                        k[i+1] =k[i]
                        E_cost[i] = 0
                elif dr == 3:
                    if demand_projections[i]> k[i] and demand_projections[i-1]> k[i-1] and demand_projections[i-2]> k[i-2]:
                        k[i + 1] = k[i] + ft*n0                                   
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                       k[i+1] =k[i] 
                       E_cost[i] = 0
                elif dr == 4:
                    if demand_projections[i]> k[i] and demand_projections[i-1]> k[i-1] and demand_projections[i-2]> k[i-2] and demand_projections[i-3]> k[i-3]  :
                        k[i + 1] = k[i] + ft*n0                                    
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                        k[i+1]=k[i]
                        E_cost[i] = 0
                else:
                    k[i+1]=k[i]
                    E_cost[i] = 0
            else :
                k[i+1]=k[i]
                E_cost[i] = 0      
        E_cost[T]=0

        
        for i in range(0,T+1):
            if i ==T:
                fixed_costs[i] = 0 #no leasing paid in last year
            elif i==0:
                fixed_costs[i] =  cl
                E_cost[i] = cc_initial
            else : 
                fixed_costs[i] = cl # leasing paid all years including 0CF
            
        for i in range(0,T+1) :
            revenue[i] = np.minimum(demand_projections[i], k[i])*p
            opex[i] = k[i] * cr
            CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
            NPV += CF[i]/((1+r)**i)
            model = pd.concat([CF, k, demand_projections,  E_cost ], axis = 1)
        return NPV

def NPV_garage_GA_DR(a,  demand = ''):
        a.reshape((-1,1))
        flex = round(a[0])
        a1_4= round(a[1])
        a9_12= round(a[2])
        a17_20= round(a[3])
        dr = round(a[4])
        ft= round(a[5])
        f0 = round(a[6])
        n0 = 200
        kmax = 1800
        NPV = 0
        rD0s = np.random.random_sample() # Realised demand in year 0
        rD10s = np.random.random_sample() # Realised additional demand by year 10
        rDfs =np.random.random_sample()# Realised additional demand after year 10


    
        cc_initial = cc_start(f0)        
        
        years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
        k = pd.Series(index=years , name = 'capacity')
        revenue = pd.Series(index=years)
        CF = pd.Series(index=years, name = ' Cash flow')
        demand_projections = pd.Series(index=years , name = 'demand')
        opex = pd.Series(index=years)
        fixed_costs = pd.Series(index=years)
        E_cost = pd.Series(index=years, name = ' expansion cost')
        plan=pd.Series(index=years) #this shows whether expansion is allowed each year
        
        for i in range(0,T+1):
            plan[i] = 1
            if i ==0:
                k[i] = 0 #initialize capacity to 0
            else : 
                k[i] = f0*n0  # initialize all others to initial
            for i in range(1,5):
                plan[i] = a1_4
            for i in range(9,13):
                plan[i] = a9_12   
            for i in range(17,21):
                plan[i] = a17_20
                
        
        for i in range(1,T+1): #initializing all ks to initial capacity
            demand_projections[0] = 0
            if demand == 'stochastic':
                demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s,rDfs)
            else:
                demand_projections[i] = demand_static(i)

        
        
        for i in range(1,T):
            if flex == 1 and (k[i]+ ft*n0) < kmax and plan[i] ==1:
                if dr == 1:
                    if demand_projections[i]> k[i] :
                        k[i + 1] = k[i] + ft*n0                                    #capacity added after one year
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                        k[i+1]=k[i]
                        E_cost[i] = 0
                elif dr == 2:
                    if demand_projections[i]> k[i] and demand_projections[i-1]> k[i-1] :
                        k[i + 1] = k[i] + ft*n0                                    #capacity added after two years
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                        k[i+1] =k[i]
                        E_cost[i] = 0
                elif dr == 3:
                    if demand_projections[i]> k[i] and demand_projections[i-1]> k[i-1] and demand_projections[i-2]> k[i-2]:
                        k[i + 1] = k[i] + ft*n0                                   
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                       k[i+1] =k[i] 
                       E_cost[i] = 0
                elif dr == 4:
                    if demand_projections[i]> k[i] and demand_projections[i-1]> k[i-1] and demand_projections[i-2]> k[i-2] and demand_projections[i-3]> k[i-3]  :
                        k[i + 1] = k[i] + ft*n0                                    
                        E_cost[i] =  Exp_cost(k[i], ft)
                    else:
                        k[i+1]=k[i]
                        E_cost[i] = 0
                else:
                    k[i+1]=k[i]
                    E_cost[i] = 0
            else :
                k[i+1]=k[i]
                E_cost[i] = 0      
        E_cost[T]=0

        
        for i in range(0,T+1):
            if i ==T:
                fixed_costs[i] = 0 #no leasing paid in last year
            elif i==0:
                fixed_costs[i] =  cl
                E_cost[i] = cc_initial
            else : 
                fixed_costs[i] = cl # leasing paid all years including 0CF
            
        for i in range(0,T+1) :
            revenue[i] = np.minimum(demand_projections[i], k[i])*p
            opex[i] = k[i] * cr
            CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
            NPV += CF[i]/((1+r)**i)
            model = pd.concat([CF, k, demand_projections,  E_cost ], axis = 1)
        return NPV, model


def CDF_GA_DR(nsim, DV):
    ENPV_res =[]
    for i in range(nsim):
        ENPV2, model2 = NPV_garage_GA_DR(DV, demand = 'stochastic') 
        ENPV_res.append(ENPV2)
    EPV = np.array(ENPV_res)
    ENPV = np.mean(ENPV_res)
    fig, ax = plt.subplots(figsize=(8, 4))
    cdf = ax.hist(EPV, 100, density=True, histtype='step',
                       cumulative=True, label='2000 simulations')
    plt.axvline(ENPV, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(ENPV*1.1, max_ylim*0.9, 'ENPV: {:.2f}'.format(ENPV))
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_title('CDF of design solution: GA with full decision rules')
    ax.set_xlabel('NPV of plan($)')
    ax.set_ylabel('Probability')  
    return cdf


def ENPV_MC_DR(nsim, DV):
    ENPV_res =[]
    for i in range(nsim):
        ENPV2, model2 = NPV_garage_GA_DR(DV, demand = 'stochastic')
        ENPV_res.append(ENPV2)
    ENPV = np.mean(ENPV_res)   
    return ENPV

def CDF_DP_DR_fixed(nsim, policy_d,DV, f0):
    NPVd =[] # this holds results from DP policy
    NPVs =[] # this holds results for GA DR approach
    NPVf =[] # this holds results for inflexible baseline
    for i in range(nsim):
        NPV_d , model_d = NPV_garage_DP_BI(policy_d, demand = 'stochastic') 
        NPV_s, models  = NPV_garage_GA_DR(DV, demand = 'stochastic')
        NPV_f  = NPV_garage_inflex(f0, demand = 'stochastic') 
        NPVd.append(NPV_d)
        NPVs.append(NPV_s)
        NPVf.append(NPV_f)
    EPVd = np.array(NPVd)
    EPVs = np.array(NPVs)
    EPVf = np.array(NPVf)
    ENPVd = np.mean(EPVd)
    ENPVs = np.mean(EPVs)         
    ENPVf = np.mean(EPVf)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_d = bx.hist(EPVd, 100, density=True, histtype='step',
                        cumulative=True, label='DP policy')
    cdf_s = bx.hist(EPVs, 100, density=True, histtype='step',
                        cumulative=True, label='GA DR')
    cdf_f = bx.hist(EPVf, 100, density=True, histtype='step',
                        cumulative=True, label='Inflexible baseline')    
    plt.axvline(ENPVd, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.25, 'ENPV DP: {:.2f} Million $'.format(ENPVd/1000000))
    plt.axvline(ENPVs, color='darkorange', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.6, 'ENPV  GA DR: {:.2f} Million $'.format(ENPVs/1000000))
    plt.axvline(ENPVf, color='darkgreen', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.45, 'ENPV  inflexible: {:.2f} Million $'.format(ENPVf/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions : DP vs GA DR vs Inflexible  comparison')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_d

def CDF_DP_SA_DR_fixed(nsim, scenario_df , policy_d, DV, f0):
    NPVd =[] # this holds results from DP policy
    NPVs =[] # this holds results for GA DR approach
    NPVf =[] # this holds results for inflexible baseline
    for i in range(nsim):
        NPV_d , model_d = NPV_garage_DP_BI(policy_d, demand = 'stochastic') 
        NPV_s, models  = NPV_garage_GA_DR_scenarios(DV, scenario_df, nsim, demand = 'stochastic')
        NPV_f  = NPV_garage_inflex(f0, demand = 'stochastic') 
        NPVd.append(NPV_d)
        NPVs.append(NPV_s)
        NPVf.append(NPV_f)
    EPVd = np.array(NPVd)
    EPVs = np.array(NPVs)
    EPVf = np.array(NPVf)
    ENPVd = np.mean(EPVd)
    ENPVs = np.mean(EPVs)         
    ENPVf = np.mean(EPVf)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_d = bx.hist(EPVd, 100, density=True, histtype='step',
                        cumulative=True, label='DP policy')
    cdf_s = bx.hist(EPVs, 100, density=True, histtype='step',
                        cumulative=True, label='GA DR')
    cdf_f = bx.hist(EPVf, 100, density=True, histtype='step',
                        cumulative=True, label='Inflexible baseline')    
    plt.axvline(ENPVd, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.25, 'ENPV DP: {:.2f} Million $'.format(ENPVd/1000000))
    plt.axvline(ENPVs, color='darkorange', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.6, 'ENPV  GA DR: {:.2f} Million $'.format(ENPVs/1000000))
    plt.axvline(ENPVf, color='darkgreen', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.45, 'ENPV  inflexible: {:.2f} Million $'.format(ENPVf/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions : DP vs GA DR vs Inflexible  comparison')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_d