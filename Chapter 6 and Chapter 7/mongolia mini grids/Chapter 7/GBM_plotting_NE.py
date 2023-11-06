# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:00:33 2022

@author: cesa_
"""



import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy import *
from matplotlib.ticker import PercentFormatter

def gbm_sim_det (s0, mu, sigma, T, dt):
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = s0*np.exp(X)
    return S

def gbm_sim (s0, mu, sigma, T, dt):
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = s0*np.exp(X) ### geometric brownian motion ###
    return S


def gbm_sim_series (s0, mu, sigma, T, dt):
    values = []
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = s0*np.exp(X) ### geometric brownian motion ###
    values.append(S)
    return values

#### carbon pricing example ####

china_cc_20 = 7
china_cc_50 = 26
years = 31

china_g = (china_cc_50 - china_cc_20) / (china_cc_20* years)

########################## DATA GENERATION #######

SCENARIOS = 2000
conf_level = 1.96 # 95th percentile confidence

### PC1######
pc_years = list(range(7))
vals = []
mu = 0
sigma = .3
T = 7
dt = 1
for i in range(SCENARIOS):
    #s0 = np.random.uniform(.1, .5)
    s0 = np.random.choice(np.arange(.1, .51, .05))
    pc1_sim = gbm_sim(s0, mu, sigma, T, dt)
    vals.append(pd.Series(pc1_sim))
pc1_df = pd.DataFrame(vals)
pc1 = pc1_df.mean(axis = 0)
pc1_std = pc1_df.std(axis = 0) * conf_level

#pc1_df.quantile(axis = 0 , q = .8 )

####PC2######
vals = []
mu = 0
sigma = .4
T = 7
dt = 1
for i in range(SCENARIOS):
    #s0 = np.random.uniform(.1, .5)
    s0 = np.random.choice(np.arange(.1, .81, .05))
    pc2_sim = gbm_sim(s0, mu, sigma, T, dt)
    vals.append(pd.Series(pc2_sim))
pc2_df = pd.DataFrame(vals)
pc2 = pc2_df.mean(axis = 0)
pc2_std = pc2_df.std(axis = 0) * conf_level



####PC3######
vals = []
mu = .01
sigma = .1
T = 31
dt = 1
for i in range(SCENARIOS):
    #s0 = np.random.uniform(.1, .5)
    s0 = np.random.choice(np.arange(.1, .45, .005))
    pc3_sim = gbm_sim(s0, mu, sigma, T, dt)
    vals.append(pd.Series(pc3_sim))
pc3_df = pd.DataFrame(vals)
pc3 = pc3_df.mean(axis = 0)
pc3_std = pc3_df.std(axis = 0) * conf_level


#### PC 4 ####
china_cc_20 = 6.7
china_cc_50 = 28
years = 31

china_g = (china_cc_50 - china_cc_20) / (china_cc_20* years)


vals = []
s0 = 6.7
mu = .0567
sigma = .1
T = 31
dt = 1
for i in range(SCENARIOS):
    #s0 = np.random.uniform(.1, .5)
    #s0 = np.random.choice(np.arange(.1, .5, .005))
    pc4_sim = gbm_sim(s0, mu, sigma, T, dt)
    vals.append(pd.Series(pc4_sim))
pc4_df = pd.DataFrame(vals)
pc4 = pc4_df.mean(axis = 0)
pc4_std = pc4_df.std(axis = 0) * conf_level




###### FULL FIGURE ATTEMPT ####
fig, axs = plt.subplots(2, 2, figsize =(16,12))

#PC1
pc_years = list(range(7))
y = pc1
yerr = pc1_std

lb = y - yerr
ub = y + yerr
lb.iloc[0] = .12
ub.iloc[0] = .475




axs[0,0].plot(y , color = 'green' , label = 'PC1 - Mean')
axs[0,0].fill_between(pc_years, lb, ub, color = 'green', alpha = .1)

axs[0,0].plot(pc1_df.iloc[4]*.95 , color = 'green' , label = 'PC1- Ex. 1' , ls='-.')
axs[0,0].plot(pc1_df.iloc[1998]*1.2 , color = 'green' , label = 'PC1- Ex. 2', ls='--')
axs[0,0].plot(pc1_df.iloc[1997]*.85 , color = 'green' , label = 'PC1- Ex. 3', ls=':')

axs[0,0].set_xticks(list(range(7)))
xt = list(range(2022,2053, 5))
axs[0,0].set_xticklabels(xt, fontsize = 10)
axs[0,0].set_ylabel('RE CAPEX Subsidy (%)',  size = '12')
axs[0,0].yaxis.set_major_formatter(PercentFormatter(1))
axs[0,0].set_xlim((0,6))
axs[0,0].set_ylim((0,1.05))
axs[0,0].set_title('PC1', weight="bold",  size = '14')
axs[0,0].legend(loc = 'upper left')
axs[0,0].text(-.4, 1.05, 'a', fontsize='14', va='bottom', fontfamily='serif', weight = 'bold')

#PC2####
y = pc2
yerr = pc2_std*.6


lb = y - yerr
ub = y + yerr
lb.iloc[0] = .14
ub.iloc[0] = .778
lb = lb.where(lb>0, 0)
ub = ub.where(ub<1, 1)



axs[0,1].plot(y , color = 'RED' , label = 'PC2- Mean')
axs[0,1].fill_between(pc_years, lb, ub, color = 'red', alpha = .1)

axs[0,1].plot(pc2_df.iloc[4].where(pc2_df.iloc[4]<1, 1) , color = 'red' , label = 'PC2- Ex. 1', ls='-.')
axs[0,1].plot(pc2_df.iloc[1998].where(pc2_df.iloc[1998]<1, 1) , color = 'red' , label = 'PC2- Ex. 2',ls='--')
#axs[0,1].plot((pc2_df.iloc[2] / 1.15).where(pc2_df.iloc[2]<1,1 ), color = 'red' , label = 'PC2- Ex. 3', ls=':')
axs[0,1].plot((pc2_df.iloc[1997]).where(pc2_df.iloc[2]<1,1 ), color = 'red' , label = 'PC2- Ex. 3', ls=':')



axs[0,1].set_xticks(list(range(7)))
xt = list(range(2022,2053, 5))
axs[0,1].set_xticklabels(xt, fontsize = 10)
axs[0,1].set_ylabel('EH CAPEX Subsidy (%)',  size = '12')
axs[0,1].yaxis.set_major_formatter(PercentFormatter(1))
axs[0,1].set_xlim((0,6))
axs[0,1].set_ylim((0,1.05))
axs[0,1].set_title('PC2', weight="bold",  size = '14')
axs[0,1].legend(loc = 'upper left')
axs[0,1].text(-.4, 1.05, 'b', fontsize='14', va='bottom', fontfamily='serif', weight = 'bold')


#PC3####
y = pc3
yerr = pc3_std
lb = y - yerr
ub = y + yerr
lb.iloc[0] = .11
ub.iloc[0] = .385

pc_years = list(range(31))

axs[1,0].plot(y , color = 'purple' , label = 'PC3- Mean')
axs[1,0].fill_between(pc_years, lb, ub, color = 'purple', alpha = .1)

axs[1,0].plot(pc3_df.iloc[4] , color = 'purple' , label = 'PC3- Ex. 1', ls='-.')
axs[1,0].plot(pc3_df.iloc[2] , color = 'purple' , label = 'PC3- Ex. 2',ls='--')
axs[1,0].plot(pc3_df.iloc[1999] , color = 'purple' , label = 'PC3- Ex. 3', ls=':')

axs[1,0].set_xticks(range(0,31,5))
xt = list(range(2022,2053, 5))
axs[1,0].set_xticklabels(xt, fontsize = 10)
axs[1,0].set_ylabel('RE Average Feed-in Tariff (2022 USD/ kWh)',  size = '12')
axs[1,0].set_xlim((0,30))
axs[1,0].set_ylim((0,1.05))
axs[1,0].set_title('PC3', weight="bold",  size = '14')
axs[1,0].legend(loc = 'upper left')
axs[1,0].text(-2, 1.1, 'c', fontsize='14', va='bottom', fontfamily='serif', weight = 'bold')


#PC4####
y = pc4
yerr = pc4_std
pc_years = list(range(31))

axs[1,1].plot(y , color = 'black' , label = 'PC4- Mean')
axs[1,1].fill_between(pc_years, y - yerr, y+yerr, color = 'black', alpha = .1)

axs[1,1].plot(pc4_df.iloc[1] , color = 'black' , label = 'PC4-Ex. 1', ls='-.')
axs[1,1].plot(pc4_df.iloc[2] , color = 'black' , label = 'PC4-Ex. 2',ls='--')
axs[1,1].plot(pc4_df.iloc[1998] , color = 'black' , label = 'PC4-Ex. 3', ls=':')

axs[1,1].set_xticks(list(range(0,31,5)))
xt = list(range(2022,2053, 5))
axs[1,1].set_xticklabels(xt, fontsize = 10)
axs[1,1].set_ylabel('China ETS Carbon Credit (2022 USD/ t CO2 Eq.)',  size = '12')
axs[1,1].set_xlim((0,30))
axs[1,1].set_ylim((0,102))
axs[1,1].set_title('PC4', weight="bold",  size = '14')
axs[1,1].legend(loc = 'upper left')
axs[1,1].text(-2,105, 'd', fontsize='14', va='bottom', fontfamily='serif', weight = 'bold')
