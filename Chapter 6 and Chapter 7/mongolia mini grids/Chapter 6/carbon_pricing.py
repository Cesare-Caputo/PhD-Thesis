# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:21:58 2021

@author: cesa_
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Inflexible_baseline_mongolia_monthly import *


#emission factor parametera
coal_c02_emission_factor = 1.37 # tonnes of co2 per metric tonne coal burned
avg_co2_emission_factor_grid = .7111 # tonnes co2 per mwh
diesel_c02_emission_factor = 2.62 # kgof co2 per litre diesel burned

# carbon cost profile parameters
china_cc_20 = 7
china_cc_30 = 14
china_cc_50 = 26


eu_cc_20 = 35
eu_cc_50 = 85

T = 360
years = list(range(2020,2051))
# print(years)

china_g = (china_cc_50 - china_cc_20) / T
eu_g = (eu_cc_50 - eu_cc_20) / T


def china_cc_projections(t):
    price = china_cc_20 + t*china_g
    return price




def eu_cc_projections(t): 
    price = eu_cc_20 + t*eu_g
    return price



def china_cc_series(T):
    prices = []
    for i in range(T):
        p_t = china_cc_projections(i)
        prices.append(p_t)
    return prices

def eu_cc_series(T):
    prices = []
    for i in range(T):
        p_t = eu_cc_projections(i)
        prices.append(p_t)
    return prices


# coal used assumed to be in kg
def estimate_co2_footprint(coal_used, eh_from_grid, diesel_liters ):
    tot_ghg = (coal_used * coal_c02_emission_factor * .001)  + (eh_from_grid * avg_co2_emission_factor_grid*.001) + (diesel_liters  * diesel_c02_emission_factor * .001)
    return tot_ghg


# pv0_homer = 363
# pv0_ga = 885


# inflex_design = np.array([380, 0])


# inflex_df = inflex_mongolia_df(inflex_design)

# inflex_carbon_evolution = inflex_df["CO2"]


def estimate_monthly_co2_reduction_frombaseline (monthly_co2 , time_step):
    monthly_reduction = inflex_carbon_evolution[time_step] - monthly_co2
    if monthly_reduction < 0:
        value = 0
    else: value = monthly_reduction
    return value




a = china_cc_series(361)



# print(china_cc_series(361))
# print(eu_cc_series(361))


china_proj = china_cc_series(31)
eu_proj = eu_cc_series(31)

plt.plot(years, china_proj, label = 'China')
plt.plot(years, eu_proj, label = 'EU')
# def carbon_price_projections()