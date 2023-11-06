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
module_size = 50


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
flex_benefit = .8
flex_cost = .15

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
elec_generation_rate = 230 #kwh/ton

#disposal parameters
purity = .7 #ratio
comp_gen_rate = .18
disp_rate = .05


def expansion_cost(capacity_added):
    e_cost = capex_cost_coeff *(capacity_added** (eos_factor)) * flex_benefit
    return e_cost                    



def capex_fac_decentralized(n_plants, eos, cost_coeff):
    capex = cost_coeff * ((capex_unit / n_plants) ** eos)
    return capex


def tot_capex_decentralized(n_plants , capacity):
    capex = n_plants * (capex_cost_coeff *capacity ** eos_factor)
    return capex

#print (tot_capex_decentralized(1 , 200))

def tot_capex_decentralized_s1( capacity):
    capex = (capex_cost_coeff *capacity ** eos_factor) * (1+ flex_cost)
    return capex

#print(tot_capex_decentralized_s1(200))

def disposal_cost(fw_collected, total_capacity):
    fw_processed = np.min([fw_collected, total_capacity])
    disp_cost = (fw_processed * (1 - purity + disp_rate) + (np.max([(fw_collected - total_capacity), 0]))) * 365 * disposal_fee
    return disp_cost                                            

#print(disposal_cost(306, 200))


def opex_tot(total_capacity):
    op_cost = opex* capex_cost_coeff * total_capacity **(eos_factor)
    return op_cost

#print(opex_tot(200))

def land_cost(total_capacity):
    land_rent = total_capacity  * n_plants * land_rental_cost_installed 
    return land_rent

# print(land_cost(100))


def transport_cost_s1(sector_demand , sector_capacity):
    transport_trips = np.ceil((np.min([sector_demand, sector_capacity]) / truck_capacity_per_trip) )
    transport_dist_sec = transport_trips  * dist_collect_in_sector
    mismatch = (sector_demand - sector_capacity) 
    mismatch_trips =  np.ceil(np.max([mismatch, 0])  /  truck_capacity_per_trip)
    transport_distance_out = mismatch_trips * dist_from_sec_1 * 2 
    transport_cost = (transport_dist_sec + transport_distance_out) * petrol_per_l * 365 # this is yearly
    return transport_cost


def transport_cost_s2(sector_demand , sector_capacity):
    transport_trips = np.ceil((np.min([sector_demand, sector_capacity]) / truck_capacity_per_trip) )
    transport_dist_sec = transport_trips  * dist_collect_in_sector
    mismatch = (sector_demand - sector_capacity) 
    mismatch_trips =  np.ceil(np.max([mismatch, 0])  /  truck_capacity_per_trip)
    transport_distance_out = mismatch_trips * dist_from_sec_2 * 2 
    transport_cost = (transport_dist_sec + transport_distance_out) * petrol_per_l * 365 # this is yearly
    return transport_cost


def transport_cost_s3(sector_demand , sector_capacity):
    transport_trips = np.ceil((np.min([sector_demand, sector_capacity]) / truck_capacity_per_trip) )
    transport_dist_sec = transport_trips  * dist_collect_in_sector
    mismatch = (sector_demand - sector_capacity) 
    mismatch_trips =  np.ceil(np.max([mismatch, 0])  /  truck_capacity_per_trip)
    transport_distance_out = mismatch_trips * dist_from_sec_3 * 2 
    transport_cost = (transport_dist_sec + transport_distance_out) * petrol_per_l * 365 # this is yearly
    return transport_cost

def transport_cost_s4(sector_demand , sector_capacity):
    transport_trips = np.ceil((np.min([sector_demand, sector_capacity]) / truck_capacity_per_trip) )
    transport_dist_sec = transport_trips  * dist_collect_in_sector
    mismatch = (sector_demand - sector_capacity) 
    mismatch_trips =  np.ceil(np.max([mismatch, 0])  /  truck_capacity_per_trip)
    transport_distance_out = mismatch_trips * dist_from_sec_4 * 2 
    transport_cost = (transport_dist_sec + transport_distance_out) * petrol_per_l * 365 # this is yearly
    return transport_cost

def transport_cost_s5(sector_demand , sector_capacity):
    transport_trips = np.ceil((np.min([sector_demand, sector_capacity]) / truck_capacity_per_trip) )
    transport_dist_sec = transport_trips  * dist_collect_in_sector
    mismatch = (sector_demand - sector_capacity) 
    mismatch_trips =  np.ceil(np.max([mismatch, 0])  /  truck_capacity_per_trip)
    transport_distance_out = mismatch_trips * dist_from_sec_5 * 2 
    transport_cost = (transport_dist_sec + transport_distance_out) * petrol_per_l * 365 # this is yearly
    return transport_cost

def transport_cost_s6(sector_demand , sector_capacity):
    transport_trips = np.ceil((np.min([sector_demand, sector_capacity]) / truck_capacity_per_trip) )
    transport_dist_sec = transport_trips  * dist_collect_in_sector
    mismatch = (sector_demand - sector_capacity) 
    mismatch_trips =  np.ceil(np.max([mismatch, 0])  /  truck_capacity_per_trip)
    transport_distance_out = mismatch_trips * dist_from_sec_6 * 2 
    transport_cost = (transport_dist_sec + transport_distance_out) * petrol_per_l * 365 # this is yearly
    return transport_cost

# print(transport_cost_s1(40, 200))
# print(transport_cost_s2(51, 100))
# print(transport_cost_s3(53, 100))
# print(transport_cost_s4(82, 100))