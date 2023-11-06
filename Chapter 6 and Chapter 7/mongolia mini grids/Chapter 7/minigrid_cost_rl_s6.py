# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:39:31 2020

@author: cesa_
"""
import numpy as np
import pandas as pd
from heating_generation import *
from electricity_demand import *


# Parameters for initial design
pv_capacity = 363 # W
battery_capacity = 2000 # Wh
inverter_capacity = 237 # W

battery_per_pv = battery_capacity / pv_capacity
inverter_per_W = inverter_capacity / pv_capacity # right now going for close to 1 to 1 ratio which means inverter is likely slightly oversized



# start off by assuming costs are constant throughout
T=30 #years
n_m = 12 # number months each year
Tm = 360
pv_cost = 2.640 # $/W
battery_cost = .700 # $/Wh
wind_cost = 1.991  # $/W
pv_cost = pv_cost/2
wind_cost = wind_cost/2
battery_cost = battery_cost/2
#UPDATE RES COST TO REFLECT SUBSIDY



diesel_capex = .352 # $/W assuming values from Laura are 1kw operating 5 hrs/day 
#DIESEL COST MOST LIKELY LOWER SO LOOK INTO THIS
#VALUES FAKE FOR NOW NEED TO VERIFY
diesel_cost_per_l = .17 # $/liter
diesel_consumption_per_kwh = .51 # liters per kwh
inverter_cost = .300 # $/W
mismatch_value_kwh = .3417 # $/kWh of mismatch, this is very arbitrary but just trying to explore how it impacts things
# ESNURE THIS IS CONSISTENT WITH ENVIRONMENT CREATION
annual_capacity_shortage_pct = .05 # allowable capacity shortage as input into HOMER
# electricity price in mongolia is around .04 $/kWh so heavily subsidized
# remember to include allowable annual capacity shortage

# THIS SECTION IS PARAMETERS FOR COSTS RELATED TO CONNECTING PLUG AND PLAY
lv_cable_cost_per_m = 5 # USD per meter of low voltage cable


T_pv = 25 # years lifetime assumed in homer
T_wind = 20 # years lifetime assumed in homer


# operational/maintenance cost here
# update to include actual opexes as funciton of installed capacity when running
#THESE VALUES FOR YEARLY BASIS
battery_opex = .000003 # $/kWh


#these othger values per kW so no NEED TO DIVIDE
pv_opex = .0001 
wind_opex= .0001 
inverter_opex = .00001

# ON A MONTHLY BASIS, SIMPLY DIVIDE ABOVE BY 12
battery_opex_m = battery_opex/ n_m# $/kWh
pv_opex_m = pv_opex/ n_m
wind_opex_m= wind_opex/ n_m
inverter_opex_m = inverter_opex/ n_m


# heating options technical specs here
#changing to super low value so see impact here
# eh_capex_w = 1.52 # $ per kW
eff_electric_stove = 1
avg_co2_emission_factor_grid = .7111 # tonnes Co2 per year
eh_lifetime = 1
#parameters for financial evaluation
coal_price = 40 # $/tonne
coal_price_per_kg = coal_price*.001 # $/tonne


months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241

# inflexible system cost function here as that script was giving weird bugs
#REPEATING ALL PARAMETERS AS REALLY NOT WORKING IN OTHER SCRIPT
#THIS IS THE ONLY THING TO CHANGE, THEN CAN RUN CODE FROM HERE TO GENERATE CDF
n_scenarios = 10
n_gers = 10



# parameters for electrcity generation predictions
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]

Tm = 360 #  months in 30 years


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
avg_co2_emission_factor_grid = .7111 # tonnes Co2 per mwh
eh_lifetime = 13 # years USE LATER TO LOOK AT REPLACEMENT COSTS
eh_capex_kw = 152 # $ per kW
eh_capex_w =eh_capex_kw*.001  # $ per kW


# NOW UPDATE EH COST TO REFLET SUBSIDY ON CAPEX
eh_capex_kw = eh_capex_kw*.5
eh_capex_w =eh_capex_kw*.001  # $ per kW

def battery_per_pv_inflex(pv_cap):
    ratio = 2000/363
    batt_n = np.ceil((ratio*pv_cap) / 1000)
    batt_cap = batt_n * 1000
    return batt_cap


def pv_capex_inflex(capacity):
    #this calculated cost only, setting up inflexible optimisation
    capex = pv_cost*capacity 
    return capex


def eh_capex_inflex(capacity):
    cost = eh_capex_w*capacity
    return cost


def battery_capex_inflex(capacity): # in W
    cost = battery_cost * capacity 
    return cost

def inverter_capex_inflex (inverter_capacity): # capacity expanded in W
    extra_inverter_cost = inverter_capacity * inverter_cost
    return  extra_inverter_cost


def carbon_footprint_inflex(coal_kg, grid_kwh):
    carbon = (coal_kg * coal_c02_emission_factor * .001) + (grid_kwh * avg_co2_emission_factor_grid*.01)
    return carbon


def battery_capex(capacity):
    cost = battery_cost * capacity 
    return cost
def eh_capex(capacity):
    cost = eh_capex_w*capacity
    return cost

def pv_capex(capacity):
    #this calculated cost only, setting up inflexible optimisation
    capex = pv_cost*capacity 
    return capex

def inverter_capex(inverter_capacity): # capacity expanded in W
    extra_inverter_cost = inverter_capacity * inverter_cost
    return  extra_inverter_cost
def wind_capex(capacity):
    #this calculated cost only, setting up inflexible optimisation
    capex = wind_cost*capacity 
    return capex


def yearly_opex_diesel(diesel_yearly_kwh):
    opex = diesel_yearly_kwh * diesel_consumption_per_kwh * diesel_cost_per_l
    return opex


def monthly_opex_diesel(diesel_monthly_kwh):
    opex = diesel_monthly_kwh * diesel_consumption_per_kwh * diesel_cost_per_l
    return opex


def monthly_diesel_consumption(diesel_monthly_kwh): # returns result in litres
    opex = diesel_monthly_kwh * diesel_consumption_per_kwh 
    return opex

def monthly_opex_system (pv_capacity, battery_capacity, wind_capacity, diesel_required_kwh, inverter_capacity, coal_usage):
    opex = pv_capacity*pv_opex_m  + battery_capacity*battery_opex_m + wind_capacity*wind_opex_m + inverter_capacity*inverter_opex_m + monthly_opex_diesel(diesel_required_kwh)   # check if need to multiply by .001 to make kwh here
    return opex


def yearly_opex_system (pv_capacity, battery_capacity, wind_capacity, diesel_required_kwh, inverter_capacity):
    opex = pv_capacity*pv_opex  + battery_capacity*battery_opex + wind_capacity*wind_opex + inverter_capacity*inverter_opex + yearly_opex_diesel(diesel_required_kwh)     # check if need to multiply by .001 to make kwh here
    return opex

def salvage_pv(pv_capacity, years_used): # this could be used eventually to compute option value of abandoning some PV during project
    salvage_factor_homer_pv = .9045
    salvage_value = salvage_factor_homer_pv * (pv_cost * pv_capacity) *((T_pv - years_used)/ T_pv)
    return salvage_value

def salvage_wind(wind_capacity, years_used): # this could be used eventually to compute option value of abandoning some PV during project
    salvage_factor_homer_wind = .8542 # LOOK AT THIS, RANDOM NUMBER FOR NOW
    salvage_value = salvage_factor_homer_wind * (wind_cost * wind_capacity) *((T_wind - years_used)/ T_wind)
    return salvage_value

def yearly_opex_inflexible (pv_capacity, battery_capacity):
    opex = pv_capacity*pv_opex  + battery_capacity*battery_opex
    return opex

def monthly_opex_inflexible (pv_capacity, battery_capacity, inverter_capacity):
    opex = pv_capacity*pv_opex_m  + battery_capacity*battery_opex_m  + inverter_capacity*inverter_opex_m
    return opex


def expansion_cost_pv(action): # action 1 is add 50 W
    if action ==1:
        pv_add = 500 
    else:
        pv_add = 0 
        
    cost = pv_cost * pv_add 
    return cost

def expansion_cost_battery(action): # action 1 is add 50 W
    if action ==2:
        battery_add = 500 
    else:
        battery_add = 0 
        
    cost = battery_cost * battery_add 
    return cost

def expansion_cost_wind(action): # action 1 is add 50 W
    if action ==3:
        wind_add = 500 
    else:
        wind_add = 0 
        
    cost = wind_cost * wind_add 
    return cost

def expansion_cost_diesel(action): # action 1 is add 50 W
    if action ==4:
        diesel_add = 500 
    else:
        diesel_add = 0 
        
    cost = diesel_capex*diesel_add
    return cost


def expansion_cost_eh(action): # action 1 is add 50 W
    if action ==5:
        eh_add = 500 
    else:
        eh_add = 0 
        
    cost = eh_capex_w*eh_add
    return cost


def expansion_impact_inverter (capacity_expanded_total): # capacity expanded in W
    extra_inverter_capacity = np.rint(capacity_expanded_total * inverter_per_W)
    extra_inverter_cost = extra_inverter_capacity * inverter_cost
    return extra_inverter_capacity, extra_inverter_cost


def expansion_impact_inverter_byact(action): # capacity expanded in W
    if action != 0 or 2 or 4 or 5 : 
        capacity_expanded_total = 500 # W
        extra_inverter_capacity, extra_inverter_cost = expansion_impact_inverter(capacity_expanded_total)
    else: 
        E_cost_inverter, extra_inverter_cost = 0 , 0
        extra_inverter_capacity = 0
    return  extra_inverter_cost, extra_inverter_capacity


def expansion_impact_inverter_byact2(action): # capacity expanded in W
    if action == 1 or action ==3: 
        capacity_expanded_total = 500 # W
        extra_inverter_capacity, extra_inverter_cost = expansion_impact_inverter(capacity_expanded_total)
    else: 
        E_cost_inverter, extra_inverter_cost = 0 , 0
        extra_inverter_capacity = 0
    return  extra_inverter_cost, extra_inverter_capacity

def mismatch_penalty(electricity_produced, electricity_demand):
    mismatch = electricity_demand - electricity_produced
    mismatch_pct = mismatch / electricity_demand
    mismatch_treshold_capacity_shortage = (1- annual_capacity_shortage_pct) * electricity_demand
    if mismatch_pct <= annual_capacity_shortage_pct:
        penalty = 0 
    else: 
        mismatch_overcapshortage = mismatch_treshold_capacity_shortage - electricity_produced
        penalty = mismatch_overcapshortage * mismatch_value_kwh
    return penalty



def shortage_over5pct(electricity_produced, electricity_demand):
    mismatch = electricity_demand - electricity_produced
    mismatch_pct = mismatch / electricity_demand
    mismatch_treshold_capacity_shortage = (1- annual_capacity_shortage_pct) * electricity_demand
    if mismatch_pct <= annual_capacity_shortage_pct:
        mismatch_overcapshortage = 0 
    else: 
        mismatch_overcapshortage = mismatch_treshold_capacity_shortage - electricity_produced
    return mismatch_overcapshortage


def cable_CAPEX(cable_lenght):
    cost = cable_lenght * lv_cable_cost_per_m
    return cost
    

