# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:00:50 2021

@author: cesa_
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt 
import random
import numpy as np
from math import pi


# def energy_simulation(pv_capacity, wind_capacity, battery_capacity, )























# parameters for cable to determine losses
# resistivity values given at 100C as this is industry standard, starting with 20C value to see what losses are like

cable_cs = 1.5 # mm^2
phase_to_neutral_voltage = 220
phase_to_phase_voltage = phase_to_neutral_voltage*(3**.5)
resistivity_copper_20C = .017 # ohm mm2 per m
resistivity_copper_100C = .023 # ohm mm2 per m
b_1ph = 2
b_3ph = 1

#ASSUME 3 PHASE CONNECTION AMONG GERS for now

# ASSUME POWER FACTOR OF 1 WHICH IS NOT REALISTIC BUT THIS IS SIMPLIFIED ANALYSIS


#cable cost parameters, look for more realistic later on 
# cable_cost = 1 # USD per meter

def wire_resistance_1ph(lenght_cable):
    R = b_1ph * resistivity_copper_20C * lenght_cable / cable_cs
    return R


def cable_energy_losses_1ph(power_transmitted, lenght_cable):
    current = power_transmitted/ phase_to_neutral_voltage
    resistance = wire_resistance_1ph(lenght_cable)
    e_loss = resistance *(current**2)
    return e_loss


def wire_resistance_3ph(lenght_cable):
    R = b_3ph * resistivity_copper_20C * lenght_cable / cable_cs
    return R


def cable_energy_losses_3ph(power_transmitted, lenght_cable):
    current = power_transmitted/ (phase_to_phase_voltage * (3**.5))
    resistance = wire_resistance_3ph(lenght_cable)
    e_loss = 3* resistance *(current**2)
    return e_loss

def cable_energy_losses_3ph_pct(power_transmitted, lenght_cable):
    losses = cable_energy_losses_3ph(power_transmitted, lenght_cable)
    pct_loss = losses/ power_transmitted
    return pct_loss




def cable_lenght_from_radius(radius): # simple assumption that longest lenght will be through circumference of area
    lenght = 2 * pi * radius 
    return lenght

def check_additional_cabling_reqs(cluster_radius, cable_available):
    total_lenght_cabling_req = cable_lenght_from_radius(cluster_radius)# this is with assumption all gers are on outside edge so overestimating here
    if cable_available >= total_lenght_cabling_req:
        cable_extra = 0
    else:
        cable_extra = total_lenght_cabling_req - cable_available
    return cable_extra


def extra_cabling_cost(extra_lenght, cost_per_m):
    return extra_lenght * cost_per_m



def load_shedding_simulation():
    monthly_load_shed_amount = np.random.beta(1, 9)
    return monthly_load_shed_amount    




# p_test = 400 # W
# l_test = 5000 # meters

# print (cable_energy_losses_1ph(p_test, l_test))
# print (cable_energy_losses_3ph(p_test, l_test))
# print (cable_energy_losses_3ph_pct(p_test, l_test))
    
