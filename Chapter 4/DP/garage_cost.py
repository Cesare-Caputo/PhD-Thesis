# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:39:31 2020

@author: cesa_
"""

# Parameters
T=20 #years
cc = 16000# Construction cost per parking space
cl = 3600000# Annual leasing land cost
cp = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth. The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
cr = 2000# Operating cost per parking space
cct = []# Total construction cost
gc = 0.10# Growth in construction cost per floor above two floors
n0 = 200# Initial number of parking space per floor
p = 10000# Price per parking space
r = 0.12# Discount rate
fmin = 2# Minimum number of floors built
fmax = 8# Maximum number of floors built
kmax = fmax*n0

def Exp_cost(k, ft):
    Ex_cost = n0*cc*((((1+gc)**(ft))-1)/(gc))*((1+gc)**((k/n0)-1))
    return Ex_cost



def opex(k):
    return cr*k