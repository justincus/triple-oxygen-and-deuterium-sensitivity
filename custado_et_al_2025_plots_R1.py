# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:29:21 2025

@author: mcustado

This script generates Figure 2 in "Contrasting controls on δ18O-δ2H and δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)
"""

import numpy as np
import pandas as pd
import lake_balance_functions_R1 as lbf
import matplotlib.pyplot as plt

#%% Load masterlist

df_ = pd.read_csv(r'...\\bl_d17o_datasheet.csv')
df_['Sample_Collection_Date'] = pd.to_datetime(df_['Sample_Collection_Date'])
mask = ~np.isnan(df_['dD'])
df = df_[mask]

# -------------------------------------------------------------------------------------
# Load input isotope data (generated from custado_et_al_2025_input_isotopes)
# -------------------------------------------------------------------------------------

in_iso = pd.read_csv(r'...\\isotopic_inputs.csv') # Load isotopic_inputs.csv (output from custado_et_al_2025_summary_samples.py)
out_iso = pd.read_csv(r'...\\calc_outputs.csv') # Load calc_outputs.csv (output from custado_et_al_2025_mc_output_R1.py)
cl = pd.read_csv(r'...\\monthly_clim_parameters.csv') # Load monthly climatological parameters

# Separate parameters
evap_rate = cl['evap_kg_m2']
temp_mo = cl['ave_temp_C']
prec_amt = cl['prec_amt_kg_m2']

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
temp = np.sum(temp_mo*evap_rate)/np.sum(evap_rate) # Get evaporation flux-weighted annual mean temperature
hum_18 = out_iso.loc[out_iso['variable']=='h_d18O_median']['value'].values[0]
hum_17 = out_iso.loc[out_iso['variable']=='h_d17O_median']['value'].values[0]
x_18 = out_iso.loc[out_iso['variable']=='x_d18O_mean']['value'].values[0]
x_17 = out_iso.loc[out_iso['variable']=='x_d17O_mean']['value'].values[0]

#%% Retrieve isotopic composition of mass balance components

# Retrieve data from in_iso dataframe
influx_d18O = in_iso.loc[in_iso['variable']=='influx_d18O']['value'].values[0]
influx_D17O = in_iso.loc[in_iso['variable']=='influx_D17O']['value'].values[0]
influx_dD =  in_iso.loc[in_iso['variable']=='influx_dD']['value'].values[0]
influx_d_exc = in_iso.loc[in_iso['variable']=='influx_d_exc']['value'].values[0]

influx_d18O_unc = in_iso.loc[in_iso['variable']=='influx_d18O']['uncertainty'].values[0]
influx_D17O_unc = in_iso.loc[in_iso['variable']=='influx_D17O']['uncertainty'].values[0]
influx_dD_unc =  in_iso.loc[in_iso['variable']=='influx_dD']['uncertainty'].values[0]
influx_d_exc_unc = in_iso.loc[in_iso['variable']=='influx_d_exc']['uncertainty'].values[0]

lake_d18O = in_iso.loc[in_iso['variable']=='lake_d18O']['value'].values[0]
lake_D17O = in_iso.loc[in_iso['variable']=='lake_D17O']['value'].values[0]
lake_dD =  in_iso.loc[in_iso['variable']=='lake_dD']['value'].values[0]
lake_d_exc = in_iso.loc[in_iso['variable']=='lake_d_exc']['value'].values[0]

lake_d18O_unc = in_iso.loc[in_iso['variable']=='lake_d18O']['uncertainty'].values[0]
lake_D17O_unc = in_iso.loc[in_iso['variable']=='lake_D17O']['uncertainty'].values[0]
lake_dD_unc =  in_iso.loc[in_iso['variable']=='lake_dD']['uncertainty'].values[0]
lake_d_exc_unc = in_iso.loc[in_iso['variable']=='lake_d_exc']['uncertainty'].values[0]

# Retrieve data from out_iso dataframe
atm_d18O = out_iso.loc[out_iso['variable']=='atm_d18O_median']['value'].values[0]
atm_d17O = out_iso.loc[out_iso['variable']=='atm_d17O_median']['value'].values[0]
atm_dD = out_iso.loc[out_iso['variable']=='atm_dD_median']['value'].values[0]

atm_D17O = out_iso.loc[out_iso['variable']=='atm_D17O_median']['value'].values[0]
atm_d_exc = out_iso.loc[out_iso['variable']=='atm_d_exc_median']['value'].values[0]

#%% Define function to calculate lake isotopic composition

# Equation following Surma (2018), Voigt (2021), Passey and Levin (2021)

def calc_lake(h, alfa_eq, alfa_k, Ra, Ri, x):
    Rl = (alfa_eq*alfa_k*(1-h)*Ri + alfa_eq*x*h*Ra)/(alfa_eq*alfa_k*(1-h)*(1-x)+x)
    return Rl

#%% Plot theoretical evaporation lines at varying humidities in both D17O-d'18O and d-excess-d18O spaces

# Set humidity and x values
h_range = [0.4, 0.6, 0.8] # (median) humidity values to use for calculations
x_range = [0, 0.2, 0.4, 0.6, 0.8, 1] # range of Xe values to generate lines

# Set fractionation factors
a_d18O = lbf.fractionation_factor_d18O(temp) 
a_d17O = lbf.fractionation_factor_d17O(a_d18O)
a_dD = lbf.fractionation_factor_dD(temp) 

a_k_d18O_0 = 1.0285**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Merlivat 1978
a_k_d17O_0 = 1.0146**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Barkan and Luz 2007
a_k_dD_0 = 1.0251**0.5 # Kinetic fractionation at zero humidity # Merlivat 1978

# Generate plot
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,6), gridspec_kw={'width_ratios': [3.5, 4]})

## Set initial parameters
# Calculate d17 from measured D17 data
influx_d17O_ = influx_D17O/1000 + 0.528*lbf.lnrz_d(influx_d18O) # get d17 influx from D17 and d18
i_17 = 1000*(np.exp(influx_d17O_/1000)-1) # delinearize influx d17

lake_d17O_ = lake_D17O/1000 + 0.528*lbf.lnrz_d(lake_d18O) # get d17 lake from D17 and d18
l_17 = 1000*(np.exp(lake_d17O_/1000)-1) # delinearize lake d17

# Convert d (delta) values to R (isotopic ratios)
influx_R18O = lbf.convert_d_to_R(influx_d18O) 
influx_R17O = lbf.convert_d_to_R(i_17) 
influx_RD = lbf.convert_d_to_R(influx_dD) 

lake_R18O = lbf.convert_d_to_R(lake_d18O) 
lake_R17O = lbf.convert_d_to_R(i_17) 
lake_RD = lbf.convert_d_to_R(lake_dD) 

atm_R18O = lbf.convert_d_to_R(atm_d18O) 
atm_R17O = lbf.convert_d_to_R(atm_d17O) 
atm_RD = lbf.convert_d_to_R(atm_dD) 

## Loop through humidity and Xe values to generate evaporation lines
for h_ in h_range:
    
    # Initialize empty arrays for D17O excess plot 
    lake_lnrz_d18O_array = []
    lake_lnrz_d17O_array = []
    lake_D17O_array = []
    
    # Initialize empty arrays for d-excess plot
    lake_d18O_array = []
    lake_dD_array = []
    lake_d_exc_array = []
    
    for x_ in x_range:
        
        # Generate modeled lake values for each isotope
        
        Rl_18 = calc_lake(h_, a_d18O, a_k_d18O_0, atm_R18O, influx_R18O, x_)
        Rl_17 = calc_lake(h_, a_d17O, a_k_d17O_0, atm_R17O, influx_R17O, x_)
        Rl_D = calc_lake(h_, a_dD, a_k_dD_0, atm_RD, influx_RD, x_)

        dL_18_ = lbf.lnrz_R(Rl_18)
        dL_17_ = lbf.lnrz_R(Rl_17)

        DL_17 = dL_17_ - 0.528*dL_18_ # Calculate 17O-excess
        d_exc = lbf.convert_R_to_d(Rl_D) - 8*lbf.convert_R_to_d(Rl_18) # Calculate d-excess
        
        lake_lnrz_d18O_array.append(dL_18_)
        lake_lnrz_d17O_array.append(dL_17_) 
        lake_D17O_array.append(DL_17*1000)
        
        lake_d18O_array.append(lbf.convert_R_to_d(Rl_18))
        lake_d_exc_array.append(d_exc)
        
    # Plot curves in the  D17O-d18O space
    z = np.polyfit(lake_lnrz_d18O_array, lake_D17O_array, 2)
    f = np.poly1d(z)
    x_new = np.linspace(lake_lnrz_d18O_array[0], lake_lnrz_d18O_array[-1], 50)
    y_new = f(x_new)
    
    ax1.plot(x_new, y_new, ls='--', linewidth=0.5, color='black')
    ax1.scatter(lake_lnrz_d18O_array, lake_D17O_array, s=4, color='black')
    
    # Plot curves in the d-excess-d18 space
    z = np.polyfit(lake_d18O_array, lake_d_exc_array, 2)
    f = np.poly1d(z)
    x_new = np.linspace(lake_d18O_array[0], lake_d18O_array[-1], 50)
    y_new = f(x_new)
    
    ax2.plot(x_new, y_new, ls='--', linewidth=0.5, color='black')
    ax2.scatter(lake_d18O_array, lake_d_exc_array, s=4, color='black')    

## Add measured data points to the plot
# Linearize d18O for the whole dataset
lnrz_18 = df['d18O'].astype('float').apply(lambda x: lbf.lnrz_d(x))
df['lnrz_d18O'] = lnrz_18

# Inflow components data
inlet = df.loc[(df['Sample_type'] == 'Inlet')]
gw_spring = df.loc[(df['Sample_type'] == 'Ground') | (df['Sample_type'] == 'Spring')]

paris_iso = df.loc[df['Site_name'].astype(str).str.contains('Paris')]
scharles_iso = df.loc[df['Site_name'].astype(str).str.contains('St Charles Creek')]
bloom_iso = df.loc[df['Site_name'].astype(str).str.contains('Bloomington Creek')]
big_iso = df.loc[df['Site_name'].astype(str).str.contains('Big Creek')]
neden_iso = df.loc[df['Site_name'].astype(str).str.contains('North Eden Creek')]
seden_iso = df.loc[df['Site_name'].astype(str).str.contains('South Eden Creek')]
swan_iso = df.loc[df['Site_name'].astype(str).str.contains('Swan Creek')]

creeks = pd.concat([paris_iso, scharles_iso, bloom_iso, big_iso, neden_iso, seden_iso,swan_iso])

inflow = pd.concat([inlet, creeks, gw_spring])

# Lake data
lake_ = df.loc[df['Sample_type'] == 'Lake'].drop(3) # Drop mud lake data

# Retrieve mean isotopic data for different components
gw_d18O = in_iso.loc[in_iso['variable']=='gw_d18O']['value'].values[0]
creek_d18O = in_iso.loc[in_iso['variable']=='creek_d18O']['value'].values[0]
prec_d18O_amt = in_iso.loc[in_iso['variable']=='prec_d18O_amt']['value'].values[0]
inlet_d18O = in_iso.loc[in_iso['variable']=='inlet_d18O']['value'].values[0]
prec_d18O_evap = in_iso.loc[in_iso['variable']=='prec_d18O_evap']['value'].values[0]

gw_D17O = in_iso.loc[in_iso['variable']=='gw_D17O']['value'].values[0]
creek_D17O = in_iso.loc[in_iso['variable']=='creek_D17O']['value'].values[0]
prec_D17O_amt = in_iso.loc[in_iso['variable']=='prec_D17O_amt']['value'].values[0]
inlet_D17O = in_iso.loc[in_iso['variable']=='inlet_D17O']['value'].values[0]
prec_D17O_evap = in_iso.loc[in_iso['variable']=='prec_D17O_evap']['value'].values[0]

lake_d_exc = in_iso.loc[in_iso['variable']=='lake_d_exc']['value'].values[0]
influx_d_exc = in_iso.loc[in_iso['variable']=='influx_d_exc']['value'].values[0]
gw_d_exc = in_iso.loc[in_iso['variable']=='gw_d_exc']['value'].values[0]
creek_d_exc = in_iso.loc[in_iso['variable']=='creek_d_exc']['value'].values[0]
prec_d_exc_amt = in_iso.loc[in_iso['variable']=='prec_d_exc_amt']['value'].values[0]
inlet_d_exc = in_iso.loc[in_iso['variable']=='inlet_d_exc']['value'].values[0]
prec_d_exc_evap = in_iso.loc[in_iso['variable']=='prec_d_exc_evap']['value'].values[0]

# Plot lnrz_d18O vs D17O
ax1.scatter(lbf.lnrz_d(lake_d18O), lake_D17O, marker = 's', edgecolor='black', linewidth=1, s = 125, color = 'yellow', label = "Lake mean", zorder=4)
ax1.scatter(lbf.lnrz_d(influx_d18O), influx_D17O, marker = '*', edgecolor='black', linewidth=1, s = 200, color = 'yellow', label = "Inflow mean", zorder=4)
ax1.scatter(lbf.lnrz_d(gw_d18O), gw_D17O, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'violet', label = "Groundwater mean", zorder=3)
ax1.scatter(lbf.lnrz_d(creek_d18O), creek_D17O, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'orange', label = "Creek mean", zorder=3)
ax1.scatter(lbf.lnrz_d(prec_d18O_amt), prec_D17O_amt, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'purple', label = "Precipitation (mean)", zorder=3)
ax1.scatter(lbf.lnrz_d(inlet_d18O), inlet_D17O, marker = 'D', edgecolor='black', linewidth=0.5, s = 100, color = '#D55E00', label = "Inlet canal", zorder=3)

ax1.scatter(lbf.lnrz_d(prec_d18O_evap), prec_D17O_evap, marker = '^', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Precipitation (evap-flux)", zorder=3)
ax1.scatter(lbf.lnrz_d(atm_d18O), atm_D17O, marker = 'P', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Atmosphere", zorder=3)
ax1.axhline(y=32, zorder=0)

ax1.scatter(lbf.lnrz_d(prec_d18O_evap), prec_D17O_evap, marker = '^', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Precipitation (evap-flux)", zorder=3)
ax1.scatter(lbf.lnrz_d(atm_d18O), atm_D17O, marker = 'P', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Atmosphere", zorder=3)

lake_.plot(x="lnrz_d18O", y="D17O", yerr='D17O_unc', fmt ='none', elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden', alpha=0.3, ax=ax1)
lake_.plot(x="lnrz_d18O", y="D17O", kind='scatter', marker='s', edgecolor='black', linewidth=1, s = 75, color = 'gray', alpha=0.3, zorder=2, ax=ax1) # label = '_Lakes', 

ax1.axhline(y=32, zorder=0)

# Scatter plot based on elevation values
sc = ax1.scatter(inflow["lnrz_d18O"], inflow["D17O"], c=inflow["MeanElevation"], 
                cmap="PuBuGn", marker='o', edgecolor='black', linewidth=0.5,  s=75, zorder=2, label="Inflow components")

# Add error bars 
ax1.errorbar(inflow["lnrz_d18O"], inflow["D17O"], yerr=inflow["D17O_unc"],  fmt="none", elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden')
ax1.set_ylabel("∆'¹⁷O (per meg)")
ax1.set_xlabel("δ'¹⁸O (‰)")
ax1.legend_ = None

# Plot d18O vs d-excess
ax2.scatter(lake_d18O, lake_d_exc, marker = 's', edgecolor='black', linewidth=1, s = 125, color = 'yellow', label = "Lake mean", zorder=4)
ax2.scatter(influx_d18O, influx_d_exc, marker = '*', edgecolor='black', linewidth=1, s = 200, color = 'yellow', label = "Inflow mean", zorder=4)
ax2.scatter(gw_d18O, gw_d_exc, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'violet', label = "Groundwater mean", zorder=3)
ax2.scatter(creek_d18O, creek_d_exc, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'orange', label = "Creek mean", zorder=3)
ax2.scatter(prec_d18O_amt, prec_d_exc_amt, marker = '*', edgecolor='black', linewidth=0.5, s = 200, color = 'purple', label = "Precipitation (mean)", zorder=3)
ax2.scatter(inlet_d18O, inlet_d_exc, marker = 'D', edgecolor='black', linewidth=1, s = 100, color = '#D55E00', label = "Inlet canal", zorder=3)

ax2.scatter(prec_d18O_evap, prec_d_exc_evap, marker = '^', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Precipitation (evap-flux)", zorder=3)
ax2.scatter(atm_d18O, atm_d_exc, marker = 'P', edgecolor='black', linewidth=0.5, s = 125, color = 'white', hatch='/////////', label = "Atmosphere", zorder=3)
ax2.axhline(y=10, zorder=0)

lake_.plot(x="d18O", y="d_exc", yerr='d_exc_unc', fmt ='none', elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden', alpha=0.3, ax=ax2)
lake_.plot(x="d18O", y="d_exc", kind='scatter', marker='s', edgecolor='black', linewidth=0.5, s = 75, color = 'gray', label = 'Lakes', alpha=0.3, zorder=1, ax=ax2)

inflow.plot(x="d18O", y="d_exc", yerr='d_exc_unc', fmt ='none', elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden', alpha=0.3, ax=ax2)
inflow.plot(x="d18O", y="d_exc", kind='scatter', marker='*', edgecolor='black', linewidth=0.5, s = 75, color = 'gray', label = '_Inflows', alpha=0.2, zorder=1, ax=ax2)

ax2.axhline(y=10, zorder=0)

# Scatter plot based on elevation values
sc = ax2.scatter(inflow["d18O"], inflow["d_exc"], c=inflow["MeanElevation"], 
                  cmap="PuBuGn", marker='o', edgecolor='black', linewidth=0.5,  s=75, zorder=2, label="Inflow components")

# Add error bars 
ax2.errorbar(inflow["d18O"], inflow["d_exc"], yerr=inflow["d_exc_unc"],  fmt="none", elinewidth=0.5, markeredgewidth=0.5, color='black', capsize=4, zorder=1, label='_Hidden')
ax2.set_ylabel('d-excess (‰)')
ax2.set_xlabel("δ¹⁸O (‰)")

ax2.legend(loc = 'lower left')

# Add colorbar for elevation
cb  = plt.colorbar(sc)
cb.set_label("Elevation (m)")

for ax in [ax1, ax2]:
    ax.xaxis.label.set_fontsize(12)  # X-axis label font size
    ax.yaxis.label.set_fontsize(12)  # Y-axis label font size
    ax.tick_params(axis='both', labelsize=12)  # Tick label font size
    
# plt.savefig('{path}'+"fig_2.png", bbox_inches="tight", dpi=600)
plt.tight_layout()
plt.show()
