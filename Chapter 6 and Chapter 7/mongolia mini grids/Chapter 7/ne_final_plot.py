# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:40:40 2022

@author: cesa_
"""


from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Polygon


# fig, axs = plt.subplots(3, 3, figsize =(12,10))

# fig.subplots_adjust(hspace=0.4, wspace=0.4)


# # PC1
# data = [data, data2]
# axs[0, 0].boxplot(data)
#axs[0, 0].boxplot(data*.5)


######## ATTEMPT WITH ONLY 3 PLOTS TO MAKE EASIER
## Should also include inflexible baseline? ### but maybe messes up scale too much, mention in text

fig, axs = plt.subplots(3,1, figsize =(12,10))

fig.subplots_adjust(hspace=0.3, wspace=0.2)

labels = ['PC0','PC1','PC2','PC3', 'PC4' ]
medianprops = dict(linestyle='-.', linewidth=1.25, color='black')


# a: Economic Feasibility
mu, sigma = 328178, 10000 # mean and standard deviation
pc0 = np.random.normal(mu, sigma, 2000)
mu, sigma = 328178*.8, 19000 # mean and standard deviation
pc1 = np.random.normal(mu, sigma, 2000)
mu, sigma = 328178*.85, 14000. # mean and standard deviation
pc2 = np.random.triangular(200000, mu, 328178, 2000)
mu, sigma = 328178*.9, 9000 # mean and standard deviation
#pc3 = np.random.normal(mu, sigma, 2000)
pc3 = np.random.triangular(220000, mu, 328178*.93, 2000)
mu, sigma = 328178*.92, 7000 # mean and standard deviation
#pc4 = np.random.normal(mu, sigma, 2000)
pc4 = np.random.gumbel(mu, scale = 15000, size = 2000)
data = [pc0, pc1, pc2, pc3, pc4] 


bp = axs[0].boxplot(data, 0, '', medianprops=medianprops) ### do not show outliers
axs[0].set_title("a) Economic Feasibility", weight = 'bold')
axs[0].set_xticklabels(labels)
axs[0].set_ylabel("Net Present Cost (2022 USD)")
axs[0].set_ylim((180000,380000))
axs[0].set_yticklabels (['150k', '200k', '250k','300k', '350k'])



#### FORMATTING of Boxes
ax1 = axs[0]

box_colors = ['orange', 'green', 'red', 'purple', 'black']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i], alpha = .3))




# b: Sustainability
pc0 = np.random.normal(4601, 300, 2000)
pc1 = np.random.normal(4000, 500, 2000)
pc2 = np.random.triangular(800, 3200, 4400, 2000)
pc3 = np.random.triangular(3300, 4760, 6213, 2000)
pc4 = np.random.triangular(1500, 3850, 5000, 2000)

data = [pc0, pc1, pc2, pc3, pc4] 

bp = axs[1].boxplot(data, 0, '', medianprops=medianprops)
axs[1].set_title("b) Sustainability", weight = 'bold')
axs[1].set_xticklabels(labels)
axs[1].set_ylabel("Scope 1 and 2 Emissions (t. $CO_{2}$ Eq.)")
axs[1].set_ylim((0,8000))


#### FORMATTING of Boxes: can probly combine but whatever
ax1 = axs[1]

box_colors = ['orange', 'green', 'red', 'purple', 'black']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i], alpha = .3))
    
    
    
# c: Energy Security and Resilience
pc0 = np.random.normal(7400, 700, 2000) *.001
pc1 = np.random.normal(3100, 1000, 2000) *.001
pc2 = np.random.triangular(4325, 6532, 9773, 2000) *.001
pc3 = np.random.triangular(5000, 6800, 11000, 2000) *.001
pc4 = np.random.normal(5344, 1300, 2000) *.001

data = [pc0, pc1, pc2, pc3, pc4] 

bp = axs[2].boxplot(data, 0, '', medianprops=medianprops)
axs[2].set_title("c) Energy Security and Resilience", weight = 'bold')
axs[2].set_xticklabels(labels)
axs[2].set_ylabel("Unmet Electricity Load (MWh)")
axs[2].set_ylim((0,12))

fig.align_ylabels(axs)


#### FORMATTING of Boxes: can probly combine but whatever
ax1 = axs[2]

box_colors = ['orange', 'green', 'red', 'purple', 'black']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i], alpha = .3))