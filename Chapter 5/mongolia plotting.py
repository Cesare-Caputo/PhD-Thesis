# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:17:08 2021

@author: cesa_
"""

from matplotlib import pyplot as plt
import gym
from gym import envs
#import gymgarage
import os
import pandas as pd
import numpy as np

def inflex_monthly_cf_plot(demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers):
    cost_df = total_system_cost_determinstic_monthly_inflex (demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers)
    a = plt.figure()
    plt.plot(months, cost_df['Total'] , label = 'System Cost')
    plt.plot(months, cost_df['Coal'] , label = 'Coal Cost' )
    plt.plot(months, cost_df['Mismatch'] , label = 'Mismatch Cost')
    #plt.plot(months, cost_df['Opex'] , label = 'Opex')
    plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.3))
    return a