import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats



#################### expansion timings ##############################
fig, ax = plt.subplots(2, 2)
total_data =  RL_DR_hist(env, model, episodes)

### Wind
data = total_data["wind"]
ax[0,0].hist(data, density=True, bins=30, histtype='stepfilled', color = 'red', label = 'Wind')
ax[0,0].set_xticks(x_loc, labels = ['0','5', '10', '15', '20', '25', '30'])
ax[0,0].set_xlim(0,60)
ax[0,0].yaxis.set_major_formatter(PercentFormatter(1))
ax[0,0].set_xlabel('Year')
ax[0,0].set_ylabel('Percentage chosen')
ax[0,0].legend(loc = 'upper right')
y_loc = np.linspace(0,.06, 4)
ax[0,0].set_yticks(y_loc, labels = ['0.0%','2.0%', '4.0%', '6.0%'])
#### PV
data = total_data["pv"]
ax[0,1].hist(data, density=True, bins=30,  color = 'green', label = 'PV')
ax[0,1].set_xlim(0,30)
ax[0,1].yaxis.set_major_formatter(PercentFormatter(1))
ax[0,1].set_xlabel('Year')
ax[0,1].legend(loc = 'upper right')
ax[0,1].set_ylabel('Percentage chosen')
x_loc = np.linspace(0,30, 7)
ax[0,1].set_xticks(x_loc, labels = ['0','5', '10', '15', '20', '25', '30'])

#### Battery
data = total_data["wind"]
ax[1,0].hist(data, density=True, bins=30,  color = 'blue', label = 'Battery')
ax[1,0].set_xlim(0,30)
ax[1,0].yaxis.set_major_formatter(PercentFormatter(1))
ax[1,0].set_xlabel('Year')
ax[1,0].legend(loc = 'upper right')
ax[1,0].set_ylabel('Percentage chosen')
y_loc = np.linspace(0,.06, 4)
#ax[1,0].set_yticks(y_loc, labels = ['0.0%','2.0%', '4.0%', '6.0%'])
x_loc = np.linspace(0,30, 7)
ax[1,0].set_xticks(x_loc, labels = ['0','5', '10', '15', '20', '25', '30'])

#### EH
data = total_data["wind"])
ax[1,1].hist(data, density=True, bins=30,  histtype='stepfilled',color = 'orange', label = 'EH')
ax[1,1].set_xlim(0,700)
ax[1,1].yaxis.set_major_formatter(PercentFormatter(1))
ax[1,1].set_xlabel('Year')
ax[1,1].set_ylabel('Percentage chosen')
#ax[1,1].yaxis.set_major_formatter(PercentFormatter(1))
x_loc = np.linspace(0,700, 7)
ax[1,1].set_xticks(x_loc, labels = ['0','5', '10', '15', '20', '25', '30'])
y_loc = np.linspace(0,.004, 4)
ax[1,1].set_yticks(y_loc, labels = ['0.0%','2.0%', '4.0%', '6.0%'])
ax[1,1].legend(loc = 'upper right')

fig.suptitle('Distribution of Capacity Adjustment Timing by Technology',  size=12)
fig.tight_layout()
fig.subplots_adjust(top=0.88)

######################### Capacity Adjustment by magnitude #########################
fig, ax = plt.subplots(2, 2, sharey = 'all')


total_data =  RL_FT_hist(env, model, episodes)



data = total_data["wind]
bins = [0.5,1.5, 2.5,3.5]
ax[0,0].hist(data, density=True, bins=bins, histtype='bar', color = 'red',ec = 'white', linewidth = 5,align = 'left', label = 'Wind')
x_loc = [0.5, 1.5, 2.5]
ax[0,0].set_xticks(x_loc, labels = ['-500W', '+500W', r"+${\Theta_{max}}$"])
ax[0,0].yaxis.set_major_formatter(PercentFormatter(1))
ax[0,0].set_ylabel('Percentage chosen')
#ax[0,0].legend(loc = 'upper left')
ax[0,0].set_title('Wind',  size=10)


## PV
data = total_data["pv]
bins = [0.5,1.5, 2.5,3.5]
ax[0,1].hist(data, density=True, bins=bins, histtype='bar', color = 'green',ec = 'white', linewidth = 5,align = 'left', label = 'PV')
x_loc = [0.5, 1.5, 2.5]
ax[0,1].set_xticks(x_loc, labels = ['-500W', '+500W', r"+${\Theta_{max}}$"])
y_loc = [0,.2,.4,.6]
ax[0,1].set_yticks(y_loc, labels = ['0%','20%', '40%', '60%'])
ax[0,1].yaxis.set_major_formatter(PercentFormatter(1))
#ax[0,1].set_ylabel('Percentage chosen')
ax[0,1].set_title('PV',  size=10)



## Battery
data = total_data["battery]
bins = [0.5,1.5, 2.5,3.5]
ax[1,0].hist(data, density=True, bins=bins, histtype='bar', color = 'blue',ec = 'white', linewidth = 5,align = 'left', label = 'Battery')
x_loc = [0.5, 1.5, 2.5]
ax[1,0].set_xticks(x_loc, labels = ['-500Wh', '+500Wh', r"+${\Theta_{max}}$"])
ax[1,0].yaxis.set_major_formatter(PercentFormatter(1))
ax[1,0].set_ylabel('Percentage chosen')
ax[1,0].set_title('Battery',  size=10)

## EH
data = total_data["EH]
bins = [0.5,1.5, 2.5,3.5]
ax[1,1].hist(data, density=True, bins=bins, histtype='bar', color = 'orange',ec = 'white', linewidth = 5,align = 'left', label = 'EH')
x_loc = [0.5, 1.5, 2.5]
ax[1,1].set_xticks(x_loc, labels = ['-500W', '+500W', r"+${\Theta_{max}}$"])
ax[1,1].yaxis.set_major_formatter(PercentFormatter(1))
#ax[1,1].set_ylabel('Percentage chosen')
ax[1,1].set_title('EH',  size=10)



fig.suptitle('Distribution of Capacity Adjustment Magnitude by Technology',  size=12)
fig.tight_layout()
fig.subplots_adjust(top=0.88)


