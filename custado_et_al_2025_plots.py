# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:29:21 2025

@author: mcustado

This script generates the data plots in "Differing sensitivity of δ18O-δ2H versus δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)
"""

import numpy as np
import pandas as pd
import lake_balance_functions as lbf
import matplotlib.pyplot as plt

#%% Load masterlist

df_ = pd.read_csv(r'....\bl_d17o_datasheet.csv')
df_['Sample_Collection_Date'] = pd.to_datetime(df_['Sample_Collection_Date'])
mask = ~np.isnan(df_['dD'])
df = df_[mask]

# plt.rcParams['font.size'] = 15
plt.rcdefaults()

colors = {
    'blue':    '#377eb8', 
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
} 

#%% Calculate fractionation factors

## Input initial environmental conditions

temp = 11.15 # Set temperature 
hum = 0.62 # Set humidity

## Calculate equilibrium fractionation factors

a_18 = lbf.fractionation_factor_d18O(temp) 
a_17 = lbf.fractionation_factor_d17O(a_18)
a_D = lbf.fractionation_factor_dD(temp)

ep_18 = (a_18 - 1)*1000
ep_17 = (a_17 - 1)*1000
ep_D = (a_D - 1)*1000

## Calculate kinetic fractionation factors

a_k_18_0 = 1.0285**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Merlivat 1978
a_k_17_0 = 1.0146**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Barkan and Luz 2007

ep_k_18 = lbf.kinetic_en_d18O(hum) # Used for d18O-dD equations
ep_k_D = lbf.kinetic_en_dD(hum) # Used for d18O-dD equations

#%% Calculate isotopic composition of inputs to mass balance equations from the dataset
# 1. Total influx (from each inflow component)
# 2. Lake
# 3. Atmospheric moisture

# Legend:
# dX represents the isotopic parameters: d18 = d18O; d17 = d17O; dD = d2H; D17 = O-17 excess; d_exc = d-excess;
# Other symbols: _mean = mean; _unc = uncertainty

## 1. Isotopic composition of total influx (volume-weighted from each component) ##

# 1a. Precipitation data (Annual volume-weighted mean)

prec_d17_mean = -6.697106453 # vol-weighted from UVU/USU data
prec_d17_unc = 0.126446549

prec_d18_mean = -12.70086346  # vol-weighted from UVU/USU data
prec_d18_unc = 0.239171769

prec_D17_mean = 35.96243598 # vol-weighted from UVU/USU data
prec_D17_unc = 1.043432549

prec_dD_mean = -92.7600607 # vol-weighted from UVU/USU data
prec_dD_unc = 1.832156117

prec_d_exc_mean = 8.84684702 # vol-weighted from UVU/USU data
prec_d_exc_unc = 0.257565181

# 1b. Inlet canal

inlet_d17 = df.loc[df['Sample_type']=="Inlet"]['d17O'].mean()
inlet_d17_unc = df.loc[df['Sample_type']=="Inlet"]['D17O_unc_t']

inlet_d18 = df.loc[df['Sample_type']=="Inlet"]['d18O'].mean()
inlet_d18_unc = df.loc[df['Sample_type']=="Inlet"]['d18O_unc']

inlet_D17 = df.loc[df['Sample_type']=="Inlet"]['D17O'].mean()
inlet_D17_unc = df.loc[df['Sample_type']=="Inlet"]['D17O_unc_t']

inlet_D17_paired = df.loc[df['Sample_type']=="Inlet"]['D17O'][1]

inlet_dD = df.loc[df['Sample_type']=="Inlet"]['dD'].mean()
inlet_dD_unc = df.loc[df['Sample_type']=="Inlet"]['dD_unc'][1]

inlet_d_exc = df.loc[df['Sample_type']=="Inlet"]['d_exc'].mean()
inlet_d_exc_unc = df.loc[df['Sample_type']=="Inlet"]['d_exc_unc'][1]

# 1c. Groundwater + Springs

gw_d17 = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d17O'].mean()
gw_d18 = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d18O'].mean()
gw_D17 = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['D17O'].mean()
gw_dD = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['dD'].mean()
gw_d_exc = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d_exc'].mean()

gw_d17_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d17O'])
gw_d18_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d18O'])
gw_D17_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['D17O'])
gw_dD_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['dD'])
gw_d_exc_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d_exc'])

# 1d. Creeks

paris_iso = df.loc[df['Site_name'].astype(str).str.contains('Paris')]
scharles_iso = df.loc[df['Site_name'].astype(str).str.contains('St Charles Creek')]
bloom_iso = df.loc[df['Site_name'].astype(str).str.contains('Bloomington Creek')]
big_iso = df.loc[df['Site_name'].astype(str).str.contains('Big Creek')]
neden_iso = df.loc[df['Site_name'].astype(str).str.contains('North Eden Creek')]
seden_iso = df.loc[df['Site_name'].astype(str).str.contains('South Eden Creek')]
swan_iso = df.loc[df['Site_name'].astype(str).str.contains('Swan Creek')]

creeks_used_iso = pd.concat([paris_iso, scharles_iso, bloom_iso, big_iso, neden_iso, seden_iso,swan_iso])

creek_d17 = lbf.creek_wt_isotope(paris_iso['d17O'].mean(), scharles_iso['d17O'].mean(),bloom_iso['d17O'].mean(),big_iso['d17O'].mean(),neden_iso['d17O'].mean(),seden_iso['d17O'].mean(),swan_iso['d17O'])
creek_D17 = lbf.creek_wt_isotope(paris_iso['D17O'].mean(),scharles_iso['D17O'].mean(),bloom_iso['D17O'].mean(),big_iso['D17O'].mean(),neden_iso['D17O'].mean(),seden_iso['D17O'].mean(),swan_iso['D17O'])
creek_d18 = lbf.creek_wt_isotope(paris_iso['d18O'].mean(),scharles_iso['d18O'].mean(),bloom_iso['d18O'].mean(),big_iso['d18O'].mean(),neden_iso['d18O'].mean(),seden_iso['d18O'].mean(),swan_iso['d18O'])
creek_dD = lbf.creek_wt_isotope(paris_iso['dD'].mean(),scharles_iso['dD'].mean(),bloom_iso['dD'].mean(),big_iso['dD'].mean(),neden_iso['dD'].mean(),seden_iso['dD'].mean(),swan_iso['dD'])
creek_d_exc = lbf.creek_wt_isotope(paris_iso['d_exc'].mean(),scharles_iso['d_exc'].mean(),bloom_iso['d_exc'].mean(),big_iso['d_exc'].mean(),neden_iso['d_exc'].mean(),seden_iso['d_exc'].mean(),swan_iso['d_exc'])

creek_d17_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['d17O'].values, scharles_iso['d17O'].values,bloom_iso['d17O'].values,big_iso['d17O'].values,neden_iso['d17O'],seden_iso['d17O'].values,swan_iso['d17O'].values]).ravel())
creek_D17_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['D17O'].values,scharles_iso['D17O'].values,bloom_iso['D17O'].values,big_iso['D17O'].values,neden_iso['D17O'].values,seden_iso['D17O'].values,swan_iso['D17O'].values]).ravel())
creek_d18_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['d18O'].values,scharles_iso['d18O'].values,bloom_iso['d18O'].values,big_iso['d18O'].values,neden_iso['d18O'].values,seden_iso['d18O'].values,swan_iso['d18O'].values]).ravel())
creek_dD_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['dD'].values,scharles_iso['dD'].values,bloom_iso['dD'].values,big_iso['dD'].values,neden_iso['dD'].values,seden_iso['dD'].values,swan_iso['dD'].values]).ravel())
creek_d_exc_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['d_exc'].values,scharles_iso['d_exc'].values,bloom_iso['d_exc'].values,big_iso['d_exc'].values,neden_iso['d_exc'].values,seden_iso['d_exc'].values,swan_iso['d_exc'].values]).ravel())

# Get volume-weighted isotopic composition of inflow

influx_d17 = lbf.calc_wt_influx(inlet_d17, creek_d17.values[0], prec_d17_mean, gw_d17)
influx_D17 = lbf.calc_wt_influx(inlet_D17, creek_D17.values[0], prec_D17_mean, gw_D17)

influx_d18 = lbf.calc_wt_influx(inlet_d18, creek_d18.values[0], prec_d18_mean, gw_d18) 
influx_dD =  lbf.calc_wt_influx(inlet_dD, creek_dD.values[0], prec_dD_mean, gw_dD) 
influx_d_exc = lbf.calc_wt_influx(inlet_d_exc, creek_d_exc.values[0], prec_d_exc_mean, gw_d_exc) 

influx_D17_unc = lbf.uncertainty_mean([inlet_D17_unc[1],creek_D17_unc,gw_D17_unc,prec_D17_unc])
influx_dD_unc =  lbf.uncertainty_mean([inlet_dD_unc,creek_dD_unc,gw_dD_unc,prec_dD_unc]) 
influx_d_exc_unc = lbf.uncertainty_mean([inlet_d_exc_unc,creek_d_exc_unc,gw_d_exc_unc,prec_d_exc_unc]) 

## 2. Isotopic composition of the lake ##

lake_d17 = df.loc[df['Sample_type']=="Lake"]['d17O'].drop([3]).mean() # drop mud lake data
lake_d18 = df.loc[df['Sample_type']=="Lake"]['d18O'].drop([3]).mean() # drop mud lake data

lake_D17 = df.loc[df['Sample_type']=="Lake"]['D17O'].drop([3]).mean() # drop mud lake data

lake_dD = df.loc[df['Sample_type']=="Lake"].drop([3])['dD'].mean() # drop mud lake data

lake_d_exc = df.loc[df['Sample_type']=="Lake"].drop([3])['d_exc'].mean() # drop mud lake data

lake_d17_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"]['d17O'].drop([3])) # drop mud lake data
lake_d18_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"]['d18O'].drop([3])) # drop mud lake data

lake_D17_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"]['D17O'].drop([3])) # drop mud lake data
lake_dD_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"].drop([3])['dD']) # drop mud lake data

lake_d_exc_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"].drop([3])['d_exc']) # drop mud lake data

## 3. Isotopic composition of the atmospheric moisture ##

# Assume that the evaporation-flux weighted precipitation is in equilibrium 
# with atmospheic moisture (Gibson et al. 2016)

# Precipitation data (Evap-flux weighted)
prec_d17_evap = -5.857917671 # evap-flux weighted, UVU USU data
prec_d18_evap = -11.12584615 # evap-flux weighted, UVU USU data
prec_D17_evap = 35.06 # evap-flux weighted, UVU USU data
prec_dD_evap = -82.89143812 # evap-flux weighted, UVU USU data
prec_d_exc_evap = 6.115331116 # prec_dD_evap - 8*prec_d18_evap 

k = 1

atm_d18 = lbf.isotope_atm(prec_d18_evap, ep_18, k) 
atm_d17 = lbf.isotope_atm(prec_d17_evap, ep_17, k) 
atm_dD = lbf.isotope_atm(prec_dD_evap, ep_D, k) 

atm_D17 = (lbf.lnrz_d(atm_d17) - 0.528*lbf.lnrz_d(atm_d18))*1000
atm_d_exc = atm_dD - 8*atm_d18

#%% Plot theoretical evaporation lines at varying humidities in obth D17O-d'18O and d-excess-d18O spaces

h_range = [0.4, 0.6, 0.8] # humidity values to use for calculations
x_range = [0, 0.2, 0.4, 0.6, 0.8, 1] # range of Xe values to generate lines

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,6), gridspec_kw={'width_ratios': [3.5, 4]})

## Set initial parameters

# Calculate d17 from measured D17 data
influx_d17_ = influx_D17/1000 + 0.528*lbf.lnrz_d(influx_d18) # get d17 influx from D17 and d18
i_17 = 1000*(np.exp(influx_d17_/1000)-1) # delinearize influx d17

lake_d17_ = lake_D17/1000 + 0.528*lbf.lnrz_d(lake_d18) # get d17 lake from D17 and d18
l_17 = 1000*(np.exp(lake_d17_/1000)-1) # delinearize lake d17

# Convert d (delta) values to R (isotopic ratios)
Ri_18 = lbf.convert_d_to_R(influx_d18) 
Ri_17 = lbf.convert_d_to_R(i_17) 

Rl_18 = lbf.convert_d_to_R(lake_d18)
Rl_17 = lbf.convert_d_to_R(l_17)

# Calculate R of atmoshperic moisture; 
# assume ratio of evaporation flux-weighted precipitation is in equilibrium 
# with atmospheric moisture (Gibson et al. 2016)
Ra_18 = lbf.convert_d_to_R(atm_d18) 
Ra_17 = lbf.convert_d_to_R(atm_d17)

## Loop through humidity and Xe values to generate evaporation lines
for h_ in h_range:
    
    # Initialize empty arrays for 17O excess plot 
    
    lake_lnrz_18_array = []
    lake_lnrz_17_array = []
    lake_D17_array = []
    
    # Initialize empty arrays for d-excess plot
    
    lake_18_array = []
    lake_dD_array = []
    lake_d_excess_array = []
    
    ep_k_18_in = lbf.kinetic_en_d18O(h_) # Used for d18O-dD equations
    ep_k_D_in = lbf.kinetic_en_dD(h_) # Used for d18O-dD equations
    
    for z in x_range:
        
        # Generate modeled lake values for 17O excess plot
        Rl_18 = lbf.passey_and_levin(h_, a_18, a_k_18_0, Ra_18, Ri_18, z)
        Rl_17 = lbf.passey_and_levin(h_, a_17, a_k_17_0, Ra_17, Ri_17, z)
        
        dL_18_ = lbf.lnrz_R(Rl_18)
        dL_17_ = lbf.lnrz_R(Rl_17)
        
        DL_17 = dL_17_ - 0.528*dL_18_
        
        lake_lnrz_18_array.append(dL_18_)
        lake_lnrz_17_array.append(dL_17_) 
        lake_D17_array.append(DL_17*1000)
        
        # Generate modeled lake values for d-excess plot
        LS_O, l = lbf.mass_balance_ssx(h_, ep_k_18_in, ep_18, a_18, atm_d18, influx_d18, z)
        LS_D, l = lbf.mass_balance_ssx(h_, ep_k_D_in, ep_D, a_D, atm_dD, influx_dD, z)
        d_exc = LS_D - 8*LS_O
        
        lake_18_array.append(LS_O)
        lake_dD_array.append(LS_D)
        lake_d_excess_array.append(d_exc)
        
    # Plot curves in the  D17-d18 space
    z = np.polyfit(lake_lnrz_18_array, lake_D17_array, 2)
    f = np.poly1d(z)
    x_new = np.linspace(lake_lnrz_18_array[0], lake_lnrz_18_array[-1], 50)
    y_new = f(x_new)
    
    ax1.plot(x_new, y_new, ls='--', linewidth=0.5, color='black')
    ax1.scatter(lake_lnrz_18_array, lake_D17_array, s=4, color='black')
    
    # Plot curves in the d-excess-d18 space
    z = np.polyfit(lake_18_array, lake_d_excess_array, 1)
    f = np.poly1d(z)
    x_new = np.linspace(lake_18_array[0], lake_18_array[-1], 50)
    y_new = f(x_new)
    ax2.plot(x_new, y_new, ls='--', linewidth=0.5, color='black')
    ax2.scatter(lake_18_array, lake_d_excess_array, s=4, color='black')    

## Add measured data to plot

# Linearize d18O for the whole dataset
lnrz_18 = df['d18O'].astype('float').apply(lambda x: lbf.lnrz_d(x))
df['lnrz_18'] = lnrz_18

# Group the dataset between lake and inflow components
lk_ = df.loc[df['Sample_type'] == 'Lake'].drop(3)
in_ = pd.concat([(df.loc[(df['Sample_type'] == 'Inlet') | (df['Sample_type'] == 'Ground') | (df['Sample_type'] == 'Spring')])])


# Plot lnrz_d18O vs D17O
ax1.scatter(lbf.lnrz_d(lake_d18), lake_D17, marker = 's', edgecolor='black', linewidth=1, s = 125, color = 'yellow', label = "Lake mean", zorder=4)
ax1.scatter(lbf.lnrz_d(influx_d18), influx_D17, marker = '*', edgecolor='black', linewidth=1, s = 200, color = 'yellow', label = "Inflow mean", zorder=4)
ax1.scatter(lbf.lnrz_d(gw_d18), gw_D17, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'violet', label = "Groundwater mean", zorder=3)
ax1.scatter(lbf.lnrz_d(creek_d18), creek_D17, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'orange', label = "Creek mean", zorder=3)
ax1.scatter(lbf.lnrz_d(prec_d18_mean), prec_D17_mean, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'purple', label = "Precipitation (mean)", zorder=3)
ax1.scatter(lbf.lnrz_d(inlet_d18), inlet_D17, marker = 'D', edgecolor='black', linewidth=0.5, s = 125, color = '#D55E00', label = "Inlet canal", zorder=3)

ax1.scatter(lbf.lnrz_d(prec_d18_evap), prec_D17_evap, marker = '^', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Precipitation (evap-flux)", zorder=3)
ax1.scatter(lbf.lnrz_d(atm_d18), atm_D17, marker = 'P', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Atmosphere", zorder=3)

lk_.plot(x="lnrz_18", y="D17O", yerr='D17O_unc_t', fmt ='none', elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden', alpha=0.3, ax=ax1)
lk_.plot(x="lnrz_18", y="D17O", kind='scatter', marker='s', edgecolor='black', linewidth=1, s = 75, color = 'gray', label = 'Lakes', alpha=0.3, zorder=2, ax=ax1)

ax1.axhline(y=32, zorder=0)

merge = pd.concat([in_, creeks_used_iso])

# Scatter plot based on elevation values
sc = ax1.scatter(lbf.lnrz_d(merge["d18O"]), merge["D17O"], c=merge["MeanElevation"], 
                cmap="PuBuGn", marker='o', edgecolor='black', linewidth=0.5,  s=75, zorder=2, label="Inflow components")

# Add error bars 
ax1.errorbar(lbf.lnrz_d(creeks_used_iso["d18O"]), creeks_used_iso["D17O"], yerr=creeks_used_iso["D17O_unc_t"],  fmt="none", elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden')

ax1.set_ylabel("∆'¹⁷O (per meg)")
ax1.set_xlabel("δ'¹⁸O (‰)")

# Plot d18O vs d-excess

ax2.scatter(lake_d18, lake_d_exc, marker = 's', edgecolor='black', linewidth=1, s = 125, color = 'yellow', label = "_Lake mean", zorder=4)
ax2.scatter(influx_d18, influx_d_exc, marker = '*', edgecolor='black', linewidth=1, s = 200, color = 'yellow', label = "Inflow mean", zorder=4)
ax2.scatter(gw_d18, gw_d_exc, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'violet', label = "Groundwater mean", zorder=3)
ax2.scatter(creek_d18, creek_d_exc, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'orange', label = "Creek mean", zorder=3)
ax2.scatter(prec_d18_mean, prec_d_exc_mean, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'purple', label = "Precipitation (mean)", zorder=3)
ax2.scatter(inlet_d18, inlet_d_exc, marker = 'D', edgecolor='black', linewidth=1, s = 125, color = '#D55E00', label = "Inlet canal", zorder=3)

ax2.scatter(prec_d18_evap, prec_d_exc_evap, marker = '^', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Precipitation (evap-flux)", zorder=3)
ax2.scatter(atm_d18, atm_d_exc, marker = 'P', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Atmosphere", zorder=3)

lk_.plot(x="d18O", y="d_exc", yerr='d_exc_unc', fmt ='none', elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden', alpha=0.3, ax=ax2)
lk_.plot(x="d18O", y="d_exc", kind='scatter', marker='s', edgecolor='black', linewidth=0.5, s = 75, color = 'gray', label = '_Lakes', alpha=0.3, zorder=1, ax=ax2)

in_.plot(x="d18O", y="d_exc", yerr='d_exc_unc', fmt ='none', elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden', alpha=0.3, ax=ax2)
in_.plot(x="d18O", y="d_exc", kind='scatter', marker='*', edgecolor='black', linewidth=0.5, s = 75, color = 'gray', label = '_Inflows', alpha=0.2, zorder=1, ax=ax2)

ax2.axhline(y=10, zorder=0)

# Scatter plot based on elevation values
sc = ax2.scatter(merge["d18O"], merge["d_exc"], c=merge["MeanElevation"], 
                 cmap="PuBuGn", marker='o', edgecolor='black', linewidth=0.5,  s=75, zorder=2, label="_Creeks")

# Add error bars 
ax2.errorbar(creeks_used_iso["d18O"], creeks_used_iso["d_exc"], yerr=creeks_used_iso["d_exc_unc"],  fmt="none", elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden')

ax2.set_ylabel('d-excess (‰)')
ax2.set_xlabel("δ¹⁸O (‰)")

ax1.legend(loc = 'lower left')

cb  = plt.colorbar(sc)
cb.set_label("Elevation (m)")

for ax in [ax1, ax2]:
    ax.xaxis.label.set_fontsize(12)  # X-axis label font size
    ax.yaxis.label.set_fontsize(12)  # Y-axis label font size
    ax.tick_params(axis='both', labelsize=12)  # Tick label font size
# plt.savefig('{path}'+fig_2.png', bbox_inches="tight", dpi=600)
plt.tight_layout()
plt.show()