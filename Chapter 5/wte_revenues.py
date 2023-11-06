# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:32:02 2020

@author: cesa_
"""

import math
import numpy as np
import pandas as pd


# Parameters
T=30 #years
r = .08 # discount rate


#CAPEX
capex_cost_coeff = 305288 # $
capex_unit = 1400 # unit costs for plants



# cost 
p_elec = .27 # $/ kWh
p_compost = 20 # $/tonne
utilized_land_rent = 20000 # $ / month/ hecatre
reserved_land_rent = 5000 # $/month/hectare
disposal_fee = 77 #$/tonne
land_rental_cost_installed = 816 # $/tonne
land_rental_cost_reserved = 204 # $/tonne

n_plants = 6

opex = .15 # %
operation_cost_rate = .05 # %

#capacities
cap_initial = 33 # tonnnes/day
cap_max = 100 # tonnes/day
reserved_land = 3.09 # hectares

#other cost factors
eos_factor = .8

# revenues
refuse_collection_fee = 65 # $/tonne



# transportation costs and parameters
petrol_per_l = .405 # $ /l
truck_price = 100000  # $
truck_capacity_per_trip = 25 # tonnes
dist_collect_in_sector = 54 # km
# all distance values in km
dist_from_sec_1 = 0
dist_from_sec_2 = 20
dist_from_sec_3 = 25
dist_from_sec_4 = 29
dist_from_sec_5 = 36 
dist_from_sec_6 = 40


traveling_distance_per_trip = 60 # km
max_trips_per_day = 3
vehicle_daily_cost = 90 # $

# electricity generation
elec_generation_rate = 228 #kwh/ton
elec_price = .27 # $/kWh

#disposal parameters
purity = .7 #ratio
comp_gen_rate = .18
disp_rate = .05



def electricity_revenue(fw_collected, total_capacity):
    fw_processed = np.min([fw_collected, total_capacity])
    revenue = fw_processed * elec_generation_rate * elec_price * 365
    return revenue     


# print(electricity_revenue(331, 200))


def refuse_collection_revenue(fw_collected):
    ref_rev = fw_collected * 365 * refuse_collection_fee
    return ref_rev

