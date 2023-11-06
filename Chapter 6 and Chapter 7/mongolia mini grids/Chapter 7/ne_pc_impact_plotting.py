# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:02:16 2020

@author: cesa_
"""
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib
import scipy
from matplotlib.ticker import PercentFormatter

from collections import OrderedDict

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])




#### create plots, play around with size #### 
#note should probably label a-d

fig, axs = plt.subplots(2, 2, figsize =(13,11))
years = months = list(range(360))

###SDG7.1.1


###SDG7.1.1
y = pd.read_csv('sdg7.1.1_complete_v1.csv')
y.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)
#plt.plot(y)

fig, axs = plt.subplots(2, 2, figsize =(14,12))
years = months = list(range(360))

axs[0,0].plot(years, y['PC0'],  color = 'orange' , label = 'PC0', ls='-')
axs[0,0].plot(years, y['PC1'],  color = 'green' , label = 'PC1', ls='-.')
axs[0,0].plot(years, y['PC2'],  color = 'red' , label = 'PC2', ls='--')
axs[0,0].plot(years, y['PC3'],  color = 'purple' , label = 'PC3', ls=':')
axs[0,0].plot(years, y['PC4'],  color = 'black' , label = 'PC4', linestyle=linestyles['densely dashdotdotted'])

## calculate performance improvement with best option 
diff = y['PC1'] - y['PC0']
tot=diff.sum().round(1)
tot
axs[0,0].text(200, .35, ('PC1 Total Impact = '+str(tot))+ '%', color = 'green')

#Formatting and Axes parameters
ypos = np.arange(0,1.1,.1)     
yt = np.arange(.5, 1.05 , .05)
xpos= list(range(50,360,50))
#xt = list(range(0,31 ,5))
xt = list(range(2022,2053, 5))

axs[0,0].set_ylabel('SDG 7.1.1: Tier 3 Access to Electricity (%)',  size = '10')
axs[0,0].set_yticks(ypos)
axs[0,0].set_yticklabels(yt, fontsize = 10)
axs[0,0].yaxis.set_major_formatter(PercentFormatter(1))
axs[0,0].set_xticks(xpos)
axs[0,0].set_xticklabels(xt, fontsize = 10)
axs[0,0].set_xlim((50,370))
axs[0,0].set_ylim((0,1.05))
axs[0,0].fill_between(years, y['PC0'], y['PC1'], color = 'green', alpha = .3)
axs[0,0].legend(loc = 'lower right')

axs[0,0].text(0.0, 1.0, 'a', fontsize='large', va='bottom', fontfamily='serif', weight = 'bold')


#LABEL ALL###
# import matplotlib.transforms as mtransforms
# for label, ax in axs.items():
# for ax in axs:
#     label = 'a'
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
#             fontsize='medium', va='bottom', fontfamily='serif')



##### SDG 7.1.2 #### 


y = pd.read_csv('sdg7.1.2_complete_v1.csv')
y.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)



axs[0,1].plot(years, y['PC0'],  color = 'orange' , label = 'PC0', ls='-')
axs[0,1].plot(years, y['PC1'],  color = 'green' , label = 'PC1', ls='-.')
axs[0,1].plot(years, y['PC2'],  color = 'red' , label = 'PC2', ls='--')
axs[0,1].plot(years, y['PC3'],  color = 'purple' , label = 'PC3', ls=':')
axs[0,1].plot(years, y['PC4'],  color = 'black' , label = 'PC4', linestyle=linestyles['densely dashdotdotted'])

## calculate performance improvement with best option 
diff = y['PC2'] - y['PC0']
tot=diff.sum().round(1)
tot
axs[0,1].text(200, .35, ('PC2 Total Impact = '+str(tot))+ '%', color = 'red')


axs[0,1].set_ylabel('SDG 7.1.2: Access to Clean Heating (%)',  size = '10')
axs[0,1].set_yticks(ypos)
axs[0,1].set_yticklabels(yt, fontsize = 10)
axs[0,1].yaxis.set_major_formatter(PercentFormatter(1))
axs[0,1].set_xticks(xpos)
axs[0,1].set_xticklabels(xt, fontsize = 10)
axs[0,1].set_xlim((50,370))
axs[0,1].set_ylim((0,1.05))
axs[0,1].fill_between(years, y['PC0'], y['PC2'], color = 'red', alpha = .3)
axs[0,1].legend(loc = 'lower right')

axs[0,1].text(0.0, 1.0, 'b', fontsize='large', va='bottom', fontfamily='serif', weight = 'bold')


##### SDG 7.2 #### 

y = pd.read_csv('sdg7.2_complete_v1.csv')
y.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)

axs[1,0].plot(years, y['PC0'],  color = 'orange' , label = 'PC0', ls='-')
axs[1,0].plot(years, y['PC1'],  color = 'green' , label = 'PC1', ls='-.')
axs[1,0].plot(years, y['PC2']*1.02,  color = 'red' , label = 'PC2', ls='--')
axs[1,0].plot(years, y['PC3'],  color = 'purple' , label = 'PC3', ls=':')
axs[1,0].plot(years, y['PC4'],  color = 'black' , label = 'PC4', linestyle=linestyles['densely dashdotdotted'])
   
## calculate performance improvement with best option 
diff = y['PC1'] - y['PC0']
tot=diff.sum().round(1)
tot
axs[1,0].text(200, .35, ('PC1 Total Impact = '+str(tot))+ '%', color = 'green')

axs[1,0].set_ylabel('SDG 7.2:Renewable Energy (% Total Consumption)',  size = '10')
#axs[1,0].set_yticks(ypos)
axs[1,0].set_yticklabels(yt, fontsize = 10)
axs[1,0].yaxis.set_major_formatter(PercentFormatter(1))
axs[1,0].set_xticks(xpos)
axs[1,0].set_xticklabels(xt, fontsize = 10)
axs[1,0].set_xlim((50,370))
axs[1,0].set_ylim((0,1.05))
axs[1,0].fill_between(years, y['PC0'], y['PC1'], color = 'green', alpha = .3)
axs[1,0].legend(loc = 'lower right')

axs[1,0].text(0.0, 1.0, 'c', fontsize='large', va='bottom', fontfamily='serif', weight = 'bold')


##### SDG 7.b.1 #### 

y = pd.read_csv('sdg7.b.1_complete_v1.csv')
y.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)


axs[1,1].plot(years, y['PC0'],  color = 'orange' , label = 'PC0', ls='-')
axs[1,1].plot(years, y['PC1'],  color = 'green' , label = 'PC1', ls='-.')
axs[1,1].plot(years, y['PC2'],  color = 'red' , label = 'PC2', ls='--')
axs[1,1].plot(years, y['PC4'],  color = 'purple' , label = 'PC3', ls=':')
axs[1,1].plot(years, y['PC3'],  color = 'black' , label = 'PC4', 
              linestyle=linestyles['densely dashdotdotted'])


## calculate performance improvement with best option 
diff = (y['PC3'] - y['PC0']) / y['PC0']
tot=diff.sum().round(1)
tot
axs[1,1].text(200, 89, ('PC3 Total Impact = '+str(tot))+ '%', color = 'purple')

axs[1,1].set_ylabel('SDG 7.b.1:Renewable Energy Capacity (W/capita)',  size = '10')

axs[1,1].set_xticks(xpos)
axs[1,1].set_xticklabels(xt, fontsize = 10)
axs[1,1].set_xlim((50,370))
axs[1,1].fill_between(years, y['PC0'] ,y['PC4'], color = 'purple', alpha = .3)
axs[1,1].legend(loc = 'lower right')

axs[1,1].text(10, 250, 'd', fontsize='large', va='bottom', fontfamily='serif', weight = 'bold')

