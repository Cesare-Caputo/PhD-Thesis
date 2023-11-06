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
s0 = 6.7
mu = .06
sigma = .1
T = 31
dt = 1
carbon_price_sim = gbm_sim(s0, mu, sigma, T, dt)
#carbon_price_sim = gbm_sim_series (s0, mu, sigma, T, dt)
plt.plot(carbon_price_sim)
