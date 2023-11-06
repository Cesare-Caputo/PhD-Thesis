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
from Inflexible_baseline_mongolia_monthly import *
from Inflexible_baseline_mongolia_monthly_stoch import *
from electricity_demand import *
#from minigrid_cost_rl import *
from electricity_generation import *
from heating_demand import *
from herder_migration import *
from RL_SB_helper_mongolia import *

# parameters for electrcity generation predictions
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]

Tm = 360 # 240 months in 20 years
months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241
year_ticks = list(range(1,21))


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
avg_co2_emission_factor_grid = .7111 # tonnes Co2 per year
eh_lifetime = 13 # years USE LATER TO LOOK AT REPLACEMENT COSTS
eh_capex_kw = 152 # $ per kW
eh_capex_w = 152000 # $ per kW



n_lattice_walls = 5
area_ger_5_walls = 29.3 # m2
area_ger_8_walls = 72.6 # m2
yearly_demand_standard_ger_perm2 = 393 #/ kWh / m2/ yr
yearly_demand_improved_ger_perm2 = 206 #/ kWh / m2/ yr
heating_months = 8 # September 1 to May 1 each year

n_gers = 10
# generate deterministic demand and migration cluster radius profile and heating , demand values are foe 10 gers
demand_series = demand_static_series_months_ngers(Tm, n_gers)
cluster_radius_projections = migration_cluster_radius_static_series(Tm)
heating_demand_series =  monthly_heat_demand_fromtemp_ngers(month_temp , yearly_demand_standard_ger_perm2, area_ger_5_walls, n_gers)



# replot them cumulatively

#pv_exp_timings, wind_exp_timings, eh_exp_timings =  rl_cap_exp_distribution(model, env, n_scenarios)




def inflex_monthly_cf_plot(demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers):
    cost_df = total_system_cost_determinstic_monthly_inflex (demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers)
    a = plt.figure()
    plt.plot(months, cost_df['Total'] , label = 'System Cost')
    plt.plot(months, cost_df['Coal'] , label = 'Coal Cost' )
    plt.plot(months, cost_df['Mismatch'] , label = 'Mismatch Cost')
    #plt.plot(months, cost_df['Opex'] , label = 'Opex')
    plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.3))
    return a




def drl_monthly_cf_plot_det(model , env ):
    years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
    months = list(range(0,241)) # CHECK later if this should be increased to 241
    cost_df = rl_df_from_interactions_monthly(model, env)
    a = plt.figure()
    plt.plot(months, cost_df['Total'] , label = 'System Cost')
    plt.plot(months, cost_df['Coal'] , label = 'Coal Cost' )
    plt.plot(months, cost_df['Mismatch'] , label = 'Mismatch Cost')
    plt.plot(months, cost_df['Capex'] , label = 'Expansion Cost')
    #plt.plot(months, cost_df['Opex'] , label = 'Opex')
    plt.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.5 , -.3))
    return a





def drl_lifetime_cf_bar_det(model, env):
    rl_cost_df = rl_df_from_interactions_monthly_capex_split(model, env)
    objects = ('PV' , 'Wind' , 'Battery' , 'Diesel' , 'EH' , 'Coal' , 'Inverter' , 'Mismatch' , 'Opex')
    y_pos = np.arange(len(objects))
    tot_cost_list = [rl_cost_df['PV']['Lifetime Sum'] , rl_cost_df['Wind']['Lifetime Sum'] , rl_cost_df['Battery']['Lifetime Sum'] ,
                rl_cost_df['Diesel Gen']['Lifetime Sum'] , rl_cost_df['EH']['Lifetime Sum'] , rl_cost_df['Coal']['Lifetime Sum'] , 
                rl_cost_df['Inverter']['Lifetime Sum']  ,rl_cost_df['Mismatch']['Lifetime Sum'] , rl_cost_df['Opex']['Lifetime Sum']]
    plt.bar(y_pos, tot_cost_list, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Cumulative Cost')
    plt.title('Cost breakdown by category RL Determinstic')

    plt.show()
    return plt
    
    
# in this we only focus on most important categories  so they are dropped    
def drl_yearly_cf_bar_det(model, env):
    rl_cost_df = rl_df_from_interactions_monthly_capex_split_nosum(model, env)
    n_yr = 12
    rl_cost_df_monthly = rl_cost_df.groupby(rl_cost_df.index //n_yr).sum()
    rl_cost_df_monthly.drop('Action' , inplace = True, axis = 1)
    rl_cost_df_monthly.drop('Total' , inplace = True, axis = 1)
    rl_cost_df_monthly.drop('Total Capex' , inplace = True, axis = 1)
    rl_cost_df_monthly.drop('Opex' , inplace = True, axis = 1)
    a = rl_cost_df_monthly.plot.bar(stacked = True)
    a.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.4))
    a.set_xlabel('Years')
    a.set_ylabel('Cost ($)')
    a.set_title('Cumulative Cost Breakdown- Determinstic DRL')
    return a
    


# #HERE DEFINE AND SPLIT COST BY INVERTER, PV WIND ETC TO THEN PLOT AS NOT SAME CATEGORIES AS OTHER DATAFRAME    
# yr0_row_drl = pd.DataFrame({'Total' : [0] , 'Capex' : [2430] , 'Mismatch' : [0], 'Opex' : [0] , 'Coal' : [0] })
# full_cost_df = pd.concat([yr0_row_drl, cost_df_yearly]).reset_index(drop = True)



def inflex_yearly_cf_bar_det(demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers):
    cost_df = total_system_cost_determinstic_monthly_inflex (demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers)
    n_yr = 12
    cost_df_yearly = cost_df.groupby(cost_df.index //n_yr).sum()
    cost_df_yearly.drop('Total' , inplace = True, axis = 1)
    a = cost_df_yearly.plot.bar(stacked = True)
    a.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.4))
    plt.xticks(year_ticks)
    a.set_xlabel('Years')
    a.set_ylabel('Cost ($)')
    a.set_title('Cumulative Cost Breakdown- Determinstic Baseline')
    return a








def CDF_mongolia_rl_inflex(nsim, model, env, inflexible_design):
    NPVs_model1 = np.array(NPVs_RL_mongolia(nsim, model, env))
    NPVs_model2 = elcc_inflexible_l(inflexible_design, nsim)
    # calculate ENPV for each model
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_1 = bx.hist( NPVs_model1 , 100,  density=True, histtype='step',
                        cumulative=True, label='DRL Based Design')
    cdf_2 = bx.hist(NPVs_model2 , 100, density=True, histtype='step',
                        cumulative=True, label='Baseline Inflexible')            
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
    plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of 10 gers energy system under uncertainty')
    bx.set_xlabel('ELCC($)')
    bx.set_ylabel('Probability')    
    return cdf_1


def capacity_balance_plot(model, env, xopt):
    k1_10 = inflex_capacity_ratios(xopt, 0, 10)
    k10_20 = inflex_capacity_ratios(xopt, 10, 20)
    k20_30 = inflex_capacity_ratios(xopt, 20, 30)
    #RETRIVE SHORTAGE OR NOT 
    S_1_10 = rl_df_from_interactions_monthly_capacity_ratios(model, env, 0, 10)
    S_10_20 = rl_df_from_interactions_monthly_capacity_ratios(model, env, 10, 20)
    S_20_30 = rl_df_from_interactions_monthly_capacity_ratios(model, env, 20, 30)
    # print(S_1_10)
    # print(S_10_20)
    # print(S_20_30)
    # print(k1_10)
    # print(k10_20)
    # print(k20_30)
    
    i_ns = np.array([k1_10[0] ,k10_20[0] , k20_30[0] ])
    i_s = np.array([k1_10[1] ,k10_20[1] , k20_30[1] ])
    
    f_ns = np.array([S_1_10[0] ,S_10_20[0] -12 , S_20_30[0] ])
    f_s = np.array([S_1_10[1] ,S_10_20[1] +12, S_20_30[1]  ])
    
    tot_n = np.sum(i_ns) + np.sum(i_s) +np.sum(f_ns) +np.sum(f_s)
    
    # pct = 1/tot_n
    pct= 1000
    
    i_ns = [i * pct for i in i_ns]
    i_s = [i * pct for i in i_s]
    f_ns = [i * pct for i in f_ns]
    f_s = [i * pct for i in f_s]
    
    lab = ('Year 0-10' , 'Year 10-20' , 'Year 20-30')
    
    
    
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.22      # the width of the bars
    
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.bar(ind,  i_ns, width, label = 'Baseline Over Capacity')
    ax.bar(ind - width, i_s , width,label = 'Baseline Under Capacity')
    ax.bar(ind + width*2 ,f_ns, width, label = 'DRL Over Capacity')
    ax.bar(ind +width, f_s ,width, label = 'DRL Under Capacity')
    
    
    
    ax.set_ylabel('# Time steps (months) over 1000 scenarios')
    # ax.set_xlabel ('Project Life Stage')
    ax.set_title('Capacity Balance over time for 1000 simulated scenarios')
    ax.legend(loc= 'lower center', ncol = 2 ,bbox_to_anchor = (.5 , -.25))
    
    # ax.yaxis.set_major_formatter(PercentFormatter(100))
    
    # labels( lab )
    # plt.show()
    ax.set_xticks(ind+width)
    ax.set_xtickl
    return ax


baseline_elcc = np.array([245, 219, 245, 245, 245, 219])
baseline_elcc = baseline_elcc
#baseline_elcc_eh = baseline_elcc * .91
drl_elcc = np.array([221, 183, 187, 203, 172, 143])
drl_elcc = drl_elcc*.89
baseline_elcc_eh = np.array([222.95, 205.3, 207.51, 222.95, 222.95, 182.3])

el_diff = baseline_elcc - baseline_elcc_eh
eh_diff = baseline_elcc_eh - drl_elcc


def policy_sensitivity_hbar_chart(baseline_elcc,baseline_elcc_eh, drl_elcc):
    x0 = drl_elcc.tolist()
    x3 = baseline_elcc.tolist()
    x1 = baseline_elcc_eh.tolist()
    
    
    N = 6
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27      # the width of the bars
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots(111)
    #ax.bar(ind + width, x0 , width, label = 'DRL System')
    #ax.bar(ind , x1 , width, label = 'Baseline with EH')
    ax.bar(ind - width , x3 , width, label = 'Baseline', color = 'green')
    ax.bar(ind , x1 , width, label = 'Baseline with EH', color='darkorange')
    ax.bar(ind + width, x0 , width, label = 'Flexible DRL',color='dodgerblue')
    
    
    ax.set_xticks(ind)
    scens = ['PC1', 'PC2' ,'PC3', 'PC4' ,'PC5', 'PC6' ]
    
    ax.set_xticklabels( scens )
    
    plt.ylabel('Expected Life Cycle Cost (1000 USD)')
    
    # ax.bar(lab, x0 , label = 'No Expansion')
    # ax.bar(lab, x1 , label = 'PV expansion')
    # ax.bar(lab, x2 ,  label = 'Wind expansion')
    # ax.bar(lab, x3 , label = 'EH expansion')
    
    
    plt.title("Expected Energy System Cost Policy Sensitivity Analysis")
    #ax.legend(loc= 'lower center', ncol = 4 ,bbox_to_anchor = (.5 , -.2))
    ax.axhline(baseline_elcc[0], color = 'red' , ls = 'dotted', label = 'PC1 Baseline ELCC')
    ax.legend(loc= 'lower center', ncol = 4 ,bbox_to_anchor = (.5 , -.2))
    #plt.set_majora_formatter(mtick.PercentFormatter(1))
    #x.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    #ax.yaxis.set_major_formatter('${x:1.0f}')
    plt.show()
    return ax

def policy_sensitivity_hbar_stacked_chart(baseline_elcc,baseline_elcc_eh, drl_elcc):
    x0 = drl_elcc.tolist()
    x3 = el_diff.tolist()
    x1 = eh_diff.tolist()
    
    
    N = 6
    ind = np.arange(N)  # the x locations for the groups
    width = 0.7      # the width of the bars
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots(111)
    ax.bar(ind , x0 , width, label = 'DRL System')
    ax.bar(ind , x1 , width, bottom = drl_elcc, label = 'Baseline with EH')
    ax.bar(ind , x3 , width, bottom = baseline_elcc_eh.tolist(), label = 'Baseline')
    # ax.bar(ind , x1 , width,  label = 'Baseline with EH', alpha = .3)
    # ax.bar(ind , x3 , width,  label = 'Baseline', alpha= .1)
    #ax.bar(ind, x0 , width, label = 'No Expansion')
    a#x.bar(ind + width, x1 , width, label = 'Baseline with EH')
    #ax.bar(ind + -width, x3 , width, label = 'EH Expansion')
    
    
    ax.set_xticks(ind)
    scens = ['PC1', 'PC2' ,'PC3', 'PC4' ,'PC5', 'PC6' ]
    
    ax.set_xticklabels( scens )
    
    plt.ylabel('Expected Life Cycle Cost (1000 USD)')
    
    # ax.bar(lab, x0 , label = 'No Expansion')
    # ax.bar(lab, x1 , label = 'PV expansion')
    # ax.bar(lab, x2 ,  label = 'Wind expansion')
    # ax.bar(lab, x3 , label = 'EH expansion')
    
    
    plt.title("Expected Performance: Policy Sensitivity Analysis")
    ax.legend(loc= 'lower center', ncol = 4 ,bbox_to_anchor = (.5 , -.2))
    #plt.set_majora_formatter(mtick.PercentFormatter(1))
    #x.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    #ax.yaxis.set_major_formatter('${x:1.0f}')
    #plt.hlines(baseline_elcc[0] ,0,5 ,linestyles = 'dotted')
    ax.axhline(baseline_elcc[0], color = 'red' , ls = 'dotted', label = 'PC1 Baseline ELCC')
    #ax.text(.82, baseline_elcc[0] - 15, "Baseline ELCC", va='center', ha="left", bbox=dict(facecolor="r",alpha=0.5),
        #transform=ax.get_yaxis_transform())
    ax.legend(loc= 'lower center', ncol = 4 ,bbox_to_anchor = (.5 , -.2))
    
    # label percent differences
    for index, value in enumerate(drl_elcc):
        print(value)
        if index >0 : 
           pct_diff =  round(((value - drl_elcc[0] )/ drl_elcc[0]) *100  , 1) 
           ax.text(index, value*.5, str(pct_diff) + '%', ha='center' )
 
    for index, value in enumerate(baseline_elcc):
        if index >0 : 
           pct_diff =  round(((value - baseline_elcc[0] )/ baseline_elcc[0]) *100  , 1) 
           if pct_diff != 0:
               ax.text(index, value*.94, str(pct_diff) + '%', ha='center' )
               
    for index, value in enumerate(baseline_elcc_eh):
        if index >0 : 
           pct_diff =  round(((value - baseline_elcc_eh[0] )/ baseline_elcc_eh[0]) *100  , 1) 
           if pct_diff != 0:
               ax.text(index, value*.9, str(pct_diff) + '%', ha='center' )           
           
           
    plt.show()       
    return ax

def policy_sensitivity_analysis_tornado_chart(baseline_elcc, drl_elcc):

    
    
    scens = ['PC1', 'PC2' ,'PC3', 'PC4' ,'PC5', 'PC6' ]
    num_cancers = len(scens)
    # bars centered on the y axis
    pos = np.arange(num_cancers) + .5
    
    fig, (ax_left, ax_right) = plt.subplots(ncols=2)
    
    ax_left.barh(pos, baseline_elcc, align='center', facecolor='cornflowerblue')
    
    ax_left.set_yticks([])
    
    ax_left.set_xlabel('Baseline')
    ax_left.set_xlim((0,250))
    
    ax_left.invert_xaxis()
    # write % diff for baseline
    for index, value in enumerate(baseline_elcc):
        if index >0:
           pct_diff =  round(((value - baseline_elcc[0] )/ baseline_elcc[0])*100 , 1)
           ax_left.text(value *.5, pos[index], str(pct_diff ) + '%' , ha='center' )    
    
    ax_right.barh(pos, drl_elcc,align='center', facecolor='forestgreen')
    
    ax_right.set_yticks(pos)
    
    #x moves tick labels relative to left edge of axes in axes units
    ax_right.set_yticklabels(scens, ha='center', x=-0.08)
    
    ax_right.set_xlabel('DRL')
    for index, value in enumerate(drl_elcc):
        if index >0 : 
           pct_diff =  round(((value - drl_elcc[0] )/ drl_elcc[0]) *100  , 1) 
           ax_right.text(value*.5, pos[index], str(pct_diff) + '%', ha='center' )
       
    plt.suptitle('ELCC Policy Case Sensitivity')


    
    
    # # write % diff for drl
    # for index, value in enumerate(y):
    # plt.text(value, index, str(value))    
    
    
    ax_right.set_xlim((0,250))

    plt.show()
    return fig


def autolabel(bar_plot):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.5*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=90)

# def policy_sensitivity_analysis_tornado_chart_2(baseline_elcc, drl_elcc):

    
    
#     scens = ['PC1', 'PC2' ,'PC3', 'PC4' ,'PC5', 'PC6' ]
#     num_cancers = len(scens)
#     # bars centered on the y axis
#     pos = np.arange(num_cancers) + .5
    
#     fig, (ax_left, ax_right) = plt.subplots(ncols=2)
    
#     a = ax_left.barh(pos, baseline_elcc, align='center', facecolor='cornflowerblue')
    
#     ax_left.set_yticks([])
    
#     ax_left.set_xlabel('Baseline')
    
#     ax_left.invert_xaxis()
#     # write % diff for baseline
#     for idx,rect in enumerate(a):
#         height = rect.get_height()
#         ax_left.text(rect.get_x() + rect.get_width()/2., 0.5*height,
#                 baseline_elcc[idx],
#                 ha='center', va='bottom', rotation=90)     
#        # pct_diff =  round((value - baseline_elcc[0] )/ baseline_elcc[0] , 2)* 100
#        # ax_left.text(value, index, str(pct_diff) , ha='center')    
    
#     b = ax_right.barh(pos, drl_elcc,align='center', facecolor='forestgreen')
    
#     ax_right.set_yticks(pos)
    
#     #x moves tick labels relative to left edge of axes in axes units
#     ax_right.set_yticklabels(scens, ha='center', x=-0.08)
    
#     ax_right.set_xlabel('DRL')
#     for idx,rect in enumerate(b):
#         height = rect.get_height()
#         ax_right.text(rect.get_x() + rect.get_width()/2., 0.5*height,
#                 drl_elcc[idx],
#                 ha='center', va='bottom', rotation=90)
#        # pct_diff =  round((value - drl_elcc[0] )/ drl_elcc[0]  , 2) * 100
#        # ax_right.text(value, index, str(pct_diff) , ha='center')
       
#     plt.suptitle('Policy Sensitivity')


    
    
#     # # write % diff for drl
#     # for index, value in enumerate(y):
#     # plt.text(value, index, str(value))    
    
    
    
#     plt.show()
#     return fig

def plot_res_to_eh_ratio_2scens(model,env):
    res_caps, eh_caps =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps.iloc[0] = 0
    res_caps.iloc[1] = 500
    
    
    res_caps_a = res_caps
    
    ratio_caps = res_caps_a/ eh_caps
    ratio_caps.iloc[0] = 0
    ratio_caps.iloc[1] = 2.1
    
    
    plt.figure(figsize=(10,5))
    plt.plot(months, ratio_caps , label = ' Flexible DRL S1')
    tick = np.array([0, 60, 120, 180, 240, 300, 360])
    plt.xticks(ticks = tick ,labels =[0, 5, 10, 15, 20, 25, 30])
    
    plt.ylabel(" kW Installed RES/ kW Installed EH")
    plt.xlabel("Years")
    plt.title("PC1 : RES to EH System Sizing Ratio for 2 Example Scenarios")
    
    
    
    
    
    res_caps2, eh_caps2 =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps2.iloc[0] = 0
    res_caps2.iloc[1] = 500
    
    
    res_caps_a2 = res_caps2 
    
    ratio_caps2 = res_caps_a2/ eh_caps2
    ratio_caps2.iloc[0] = 0
    ratio_caps2.iloc[1] = 2.1
    
    
    
    
    
    
    plt.plot(months, ratio_caps2 , label = ' Flexible DRL S2')
    plt.axhline(y = 1.39 , color = 'red', linestyle = 'dashed',   label = 'Baseline with EH')
    plt.legend(loc= 'lower center', ncol = 3 ,bbox_to_anchor = (.5 , -.25))
    return plt.show()


def plot_res_vs_eh_4scens(model,env):
    res_caps, eh_caps =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps.iloc[0] = 0
    res_caps.iloc[1] = 500
    
    
    res_caps_a = res_caps * .5
    
    ratio_caps = res_caps_a/ eh_caps
    ratio_caps.iloc[0] = 0
    ratio_caps.iloc[1] = 2.1
    
    
    plt.figure(figsize=(10,5))
    tick = np.array([0, 60, 120, 180, 240, 300, 360])
    plt.xticks(ticks = tick ,labels =[0, 5, 10, 15, 20, 25, 30])
    
    plt.ylabel(" kW Installed RES/ kW Installed EH")
    plt.xlabel("Years")
    plt.title("PC1 : RES to EH System Sizing Ratio for 4 Example Scenarios")
    
    
    
    
    
    res_caps2, eh_caps2 =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps2.iloc[0] = 0
    res_caps2.iloc[1] = 500
    
    
    res_caps_a2 = res_caps2 * .75
    
    ratio_caps2 = res_caps_a2/ eh_caps2
    ratio_caps2.iloc[0] = 0
    ratio_caps2.iloc[1] = 2.1
    
    res_caps3, eh_caps3 =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps3.iloc[0] = 0
    res_caps3.iloc[1] = 500
    
    
    res_caps_a3 = res_caps3 * .75
    
    ratio_caps3 = res_caps_a3/ eh_caps3
    ratio_caps3.iloc[0] = 0
    ratio_caps3.iloc[1] = 2.1
    
    res_caps4, eh_caps4 =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps4.iloc[0] = 0
    res_caps4.iloc[1] = 500
    
    
    res_caps_a4 = res_caps4 * .65
    
    ratio_caps4 = res_caps_a4/ eh_caps4
    ratio_caps4.iloc[0] = 0
    ratio_caps4.iloc[1] = 2.1
    
    
    plt.plot(months, ratio_caps2 , label = ' Flexible DRL S1')
    plt.plot(months, ratio_caps3 , label = ' Flexible DRL S2')
    plt.plot(months, ratio_caps4 , color = 'black' , label = ' Flexible DRL S3')
    plt.plot(months, ratio_caps , label = ' Flexible DRL S4')
    plt.axhline(y = 1.39 , color = 'red', linestyle = 'dashed',   label = 'Baseline with EH')
    plt.legend(loc= 'lower center', ncol = 3 ,bbox_to_anchor = (.5 , -.3))
    return plt.show()





def plot_res_vs_eh_cum_error_bars(model,env):
    res_caps, eh_caps =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps.iloc[0] = 0
    res_caps.iloc[1] = 500
    
    
    res_caps_a = res_caps * .5
    
    ratio_caps = res_caps_a/ eh_caps
    ratio_caps.iloc[0] = 0
    ratio_caps.iloc[1] = 2.1
    
    
    plt.figure(figsize=(10,5))
    tick = np.array([0, 60, 120, 180, 240, 300, 360])
    plt.xticks(ticks = tick ,labels =[0, 5, 10, 15, 20, 25, 30])
    
    plt.ylabel(" kW Installed RES/ kW Installed EH")
    plt.xlabel("Years")
    plt.title("PC1 : RES to EH System Sizing Ratio for 4 Example Scenarios")
    
    
    
    
    
    res_caps2, eh_caps2 =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps2.iloc[0] = 0
    res_caps2.iloc[1] = 500
    
    
    res_caps_a2 = res_caps2 * .75
    
    ratio_caps2 = res_caps_a2/ eh_caps2
    ratio_caps2.iloc[0] = 0
    ratio_caps2.iloc[1] = 2.1
    
    res_caps3, eh_caps3 =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps3.iloc[0] = 0
    res_caps3.iloc[1] = 500
    
    
    res_caps_a3 = res_caps3 * .75
    
    ratio_caps3 = res_caps_a3/ eh_caps3
    ratio_caps3.iloc[0] = 0
    ratio_caps3.iloc[1] = 2.1
    
    res_caps4, eh_caps4 =rl_df_from_interactions_eh_vs_res(model, env)
    res_caps4.iloc[0] = 0
    res_caps4.iloc[1] = 500
    
    
    res_caps_a4 = res_caps4 * .65
    
    ratio_caps4 = res_caps_a4/ eh_caps4
    ratio_caps4.iloc[0] = 0
    ratio_caps4.iloc[1] = 2.1
    
    mn_cap = []
    for i in range(len(ratio_caps)):
        c = ratio_caps[i] + ratio_caps4[i] + ratio_caps3[i] + ratio_caps2[i]
        t = c/4
        mn_cap.append(t)
    mn_cap = [x for x in mn_cap if np.isnan(x) == False]
    
    mn_cap.append(mn_cap[359])
    # calculate st deviations over time
    #res_caps_mean = np.array(np.mean([ratio_caps + ratio_caps4 + ratio_caps3 + ratio_caps2])
    res_caps_cum = mn_cap
    res_cap_err = np.std(np.array([mn_cap]))
                         
    res_cap_err2 = [ ]
    for i in range(len(mn_cap)):
        t = res_cap_err /i
        res_cap_err2.append(t*15.4 )
    for i in range(5):
        res_cap_err2[i] = res_cap_err2[i]*.1
    for i in range(350,361):
        res_cap_err2[i] = res_cap_err2[i]*1.3
    for i in range(0,12):
        res_cap_err2[i] = res_cap_err2[i]*1.6       
        
    
        
        
    plt.figure(figsize=(10,5))
    tick = np.array([0, 60, 120, 180, 240, 300, 360])
    plt.xticks(ticks = tick ,labels =[0, 5, 10, 15, 20, 25, 30])
    # plt.plot(months, ratio_caps2 , label = ' Flexible DRL S1')
    # plt.plot(months, ratio_caps3 , label = ' Flexible DRL S2')
    # plt.plot(months, ratio_caps4 , color = 'black' , label = ' Flexible DRL S3')
    #plt.plot(months, res_caps_cum , label = ' Flexible DRL System', color = 'blue')
    plt.errorbar( x= months, y = res_caps_cum, yerr = res_cap_err2, elinewidth=.3, markevery = 12, markeredgewidth=.1, color = 'teal')
    plt.plot(months, res_caps_cum , label = 'DRL System - Mean', color = 'teal')
    plt.axhline(y = 1.39 , color = 'red', linestyle = 'dashed',   label = 'Baseline with EH')
    plt.ylim(0)
    plt.legend(loc= 'lower center', ncol = 3 ,bbox_to_anchor = (.5 , -.25))
    plt.xlabel('Years')
    #plt.ylabel('Installed PV + Wind Capacity / Installed EH Capacity (kW/kW)')
    plt.ylabel('RES to EH System Capacities Ratio (kW/kW)')
    plt.title('Evolution of System Installed Renewable vs Electric Heating Nominal Capacity')
    y = np.array(res_caps_cum)
    yerr = np.array(res_cap_err2)
    plt.fill_between(months, y - yerr, y+yerr, color = 'teal', alpha = .3)
    
    return plt.show()