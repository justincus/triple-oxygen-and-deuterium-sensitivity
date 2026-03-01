# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:52:52 2024

@author: mcustado

This script runs the Monte Carlo simulations described in "Contrasting controls on δ18O-δ2H and δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)
The parameters calculated in this file are the isotopic composition of atmospheric moisture, humidity, Xe, theta lake, and 18Orucp using Eqs. 1, 2, S-16, S-17

Variable names:
    d = small delta (δ)
    D = big delta (Δ)
    
    [sample type]_dX
    where sample type are:
        inlet = measurements from the inlet canal
        creek = measurements from the surrounding creeks
        gw = measurements from groundwater and springs
        prec = measurements from preciptiation samples from UVU and USU stations (Utah Valley University and Utah State University)
        lake = measurements from Bear Lake (Mud Lake data not included in calculations)
        influx = total volume inflow to the lake 
        atm = atmospheric moisture (estimated)
        
    Other abbreviations:
        hum = humidity
        temp = temperature
        x = Xe (evaporation-to-inflow ratio)
        mo = month
        evap = evaporation flux
        amt = amount
        dist = distribution
        unc = uncertainty
        d_exc = d-excess
        n_ = coefficienet to account for turbulence (in this work, it is set at 0.5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import lake_balance_functions_R1 as lbf

# -------------------------------------------------------------------------------------
# Load measured isotope data
# -------------------------------------------------------------------------------------

# Load sample mastersheet

df_ = pd.read_csv(r'G:\My Drive\PhD - Custado\Writing\2025_17O_BearLake\bl_d17o_datasheet.csv')
df_['Sample_Collection_Date'] = pd.to_datetime(df_['Sample_Collection_Date'])
mask = ~np.isnan(df_['dD']) # Remove samples with unpaired dD
df = df_[mask]

# Separate sample types

inlet = df.loc[df['Sample_type']=='Inlet']
creek = df.loc[df['Site_name'].astype(str).str.contains('creek|Creek')]
gw = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]
lake = df.loc[(df['Sample_type']=="Lake")]
lake = lake.drop(3) # Drop non-Bear Lake data (Mud Lake)
prec = df.loc[(df['Sample_type']=="Precipitation")]

# Get monthly average of precipitation data across years

prec['Sample_Collection_Date'] = pd.to_datetime(prec['Sample_Collection_Date'])
prec = prec.set_index('Sample_Collection_Date')

prec_d18O_mo = prec.groupby(prec.index.month)['d18O'].mean()
prec_d17O_mo = prec.groupby(prec.index.month)['d17O'].mean()
prec_dD_mo = prec.groupby(prec.index.month)['dD'].mean()
prec_D17O_mo = prec.groupby(prec.index.month)['D17O'].mean()

# -------------------------------------------------------------------------------------
# Load measured isotope data (generated from custado_et_al_2025_summary_samples)
# -------------------------------------------------------------------------------------

in_iso = pd.read_csv(r'C:\Users\mcustado\OneDrive\Python scripts for upload\lake_isotope_mass_balance\triple-oxygen-and-deuterium-sensitivity\R1\for_upload\isotopic_inputs.csv')

# -------------------------------------------------------------------------------------
# Load environmental/climate parameters 
# (extracted from North American Regional Reanalysis dataset)
# -------------------------------------------------------------------------------------

cl = pd.read_csv(r'G:\My Drive\PhD - Custado\Writing\2025_17O_BearLake\monthly_clim_parameters.csv')

# Separate parameters

evap_rate = cl['evap_kg_m2']
temp_mo = cl['ave_temp_C']
prec_amt = cl['prec_amt_kg_m2']

#%% Generate input arrays

# -------------------------------------------------------------------------------------
# Generate input distributions 
# Input variables for each function: 
    # calc_hum_17 (mass balance for triple oxygen): lake_d18O_, lake_D17O_, temp_, influx_d18O_, influx_D17O_, atm_d18O_, atm_d17O_
    # calc_hum_18 (mass balance for 18O-2H): lake_d18O_, lake_dD_, temp_, influx_d18O_, influx_D_, atm_d18O_, atm_dD_
    # to calculate atmospheric moisture isotopic composition: prec_d18, prec_d17, prec_dD
# -------------------------------------------------------------------------------------

# Set number of simulations
sim = 100000

# Initialize random number generator
rng = np.random.default_rng(seed=42) # Seed can be any integer

# Initialize function to generate distributions based on mixing of Gaussian distributions. 
# Gaussian distributions for each data point are generated using sample values and analytical uncertainties 
# See supplementary material: "Probability distributions of input parameters"
def dist_generator(data, data_unc):
    idx = rng.integers(0, len(data), size=sim) 
    dist = rng.normal(loc=data[idx], scale=data_unc[idx])
    return dist

# -------------------------------------------------------------------------------------
# Influx 
# -------------------------------------------------------------------------------------
# Generate distribution of each inflow component (1) inlet, (2) creeks, (3) precipitation, and (4) groundwater based on measured samples
# Generate inflow composition by calculating weighted means based on volumetric discharge (lbf.calc_wt_influx)

inlet_d18O_dist = dist_generator(inlet['d18O'].values, inlet['d18O_unc'].values)
creek_d18O_dist = dist_generator(creek['d18O'].values, creek['d18O_unc'].values)
gw_d18O_dist = dist_generator(gw['d18O'].values, gw['d18O_unc'].values)
prec_d18O_dist = dist_generator(prec['d18O'].values, prec['d18O_unc'].values)

influx_d18O_dist = lbf.calc_wt_influx(inlet_d18O_dist, creek_d18O_dist, prec_d18O_dist, gw_d18O_dist)

inlet_dD_dist = dist_generator(inlet['dD'].values, inlet['dD_unc'].values)
creek_dD_dist = dist_generator(creek['dD'].values, creek['dD_unc'].values)
gw_dD_dist = dist_generator(gw['dD'].values, gw['dD_unc'].values)
prec_dD_dist = dist_generator(prec['dD'].values, prec['dD_unc'].values)

influx_dD_dist = lbf.calc_wt_influx(inlet_dD_dist, creek_dD_dist, prec_dD_dist, gw_dD_dist)

inlet_D17O_dist = dist_generator(inlet['D17O'].values, inlet['D17O_unc'].values)
creek_D17O_dist = dist_generator(creek['D17O'].values, creek['D17O_unc'].values)
gw_D17O_dist = dist_generator(gw['D17O'].values, gw['D17O_unc'].values)
prec_D17O_dist = dist_generator(prec['D17O'].values, prec['D17O_unc'].values)

influx_D17O_dist = lbf.calc_wt_influx(inlet_D17O_dist, creek_D17O_dist, prec_D17O_dist, gw_D17O_dist)

# -------------------------------------------------------------------------------------
# Lake 
# -------------------------------------------------------------------------------------
lake_d18O_dist = dist_generator(lake['d18O'].values, lake['d18O_unc'].values)
lake_dD_dist = dist_generator(lake['dD'].values, lake['dD_unc'].values)
lake_D17O_dist = dist_generator(lake['D17O'].values, lake['D17O_unc'].values)

# -------------------------------------------------------------------------------------
# Precipitation (evaporation flux-weighted, to estimate isotopic composition of atmospheric moisture)
# -------------------------------------------------------------------------------------

# Calculate weights from monthly evaporation rate data
evap_weight = evap_rate.values / np.sum(evap_rate.values)

month_idx = (prec.index.month.values - 1).astype(int)

# Make d18O covary with dD and d17O based on the GMWL

# Generate input array of d18O using a weighted normal distribution
# Combine variances from sample values and their uncertainties

# Calculate weighted variance from sample values:
prec_d18O_mo_var = np.sum(evap_weight * (prec_d18O_mo - in_iso.loc[in_iso['variable']=='prec_d18O_evap']['value'].values)**2) # loc values come from isotopic inputs datasheet
prec_D17O_mo_var = np.sum(evap_weight * (prec_D17O_mo - in_iso.loc[in_iso['variable']=='prec_D17O_evap']['value'].values)**2) # loc values come from isotopic inputs datasheet
prec_dD_mo_var = np.sum(evap_weight * (prec_dD_mo - in_iso.loc[in_iso['variable']=='prec_dD_evap']['value'].values)**2) # loc values come from isotopic inputs datasheet

# Calculate weighted variance matrix from analytical uncertainties:
prec_d18O_mo_unc = prec.groupby(prec.index.month)['d18O_unc'].apply(
    lambda s: np.sqrt(np.sum(np.square(s.to_numpy()))) / len(s)
)
prec_D17O_mo_unc = prec.groupby(prec.index.month)['D17O_unc'].apply(
    lambda s: np.sqrt(np.sum(np.square(s.to_numpy()))) / len(s)
)
prec_dD_mo_unc = prec.groupby(prec.index.month)['dD_unc'].apply(
    lambda s: np.sqrt(np.sum(np.square(s.to_numpy()))) / len(s)
)

prec_d18O_mo_var_unc = np.sum(evap_weight*(prec_d18O_mo_unc**2))
prec_D17O_mo_var_unc = np.sum(evap_weight*(prec_D17O_mo_unc**2))
prec_dD_mo_var_unc = np.sum(evap_weight*(prec_dD_mo_unc**2))

# Combine generated variances:
prec_d18O_mo_std = np.sqrt(prec_d18O_mo_var + prec_d18O_mo_var_unc)
prec_D17O_mo_std = np.sqrt(prec_D17O_mo_var + prec_D17O_mo_var_unc)
prec_dD_mo_std = np.sqrt(prec_dD_mo_var + prec_dD_mo_var_unc)

# Generate d18O and D17O distribution
prec_d18O_dist = rng.normal(loc=in_iso.loc[in_iso['variable']=='prec_d18O_evap']['value'], scale=prec_d18O_mo_std, size=sim)
prec_D17O_dist = rng.normal(loc=in_iso.loc[in_iso['variable']=='prec_D17O_evap']['value'], scale=prec_D17O_mo_std, size=sim)

# Generate d17O and d18O distribution according to GMWL
prec_d17O_dist_lnrz = prec_D17O_dist/1000 + 0.528 * lbf.lnrz_d(prec_d18O_dist) # Calculate d17O from D17O data
prec_d17O_dist = 1000*(np.exp(prec_d17O_dist_lnrz/1000)-1) # Delinearize calculated d17O

offsetD = in_iso.loc[in_iso['variable']=='prec_dD_evap']['value'].values - 8 * in_iso.loc[in_iso['variable']=='prec_d18O_evap']['value'].values
prec_dD_dist = 8 * prec_d18O_dist + offsetD + np.random.normal(0, prec_dD_mo_std, sim) # Offset is to ensure mean matches measured samples

# -------------------------------------------------------------------------------------
# Temperature 
# -------------------------------------------------------------------------------------
# Generate input array using a weighted normal distribution

temp = np.sum(temp_mo*evap_rate)/np.sum(evap_rate) # Get evaporation flux-weighted annual mean temperature

temp_mo_var = np.sum(evap_weight * (temp_mo - temp)**2)
temp_mo_std = np.sqrt(temp_mo_var)
temp_dist = rng.normal(loc=temp, scale=temp_mo_std, size=sim)

# -------------------------------------------------------------------------------------
# Compile input distributions, generate histograms, print summary statistics
# -------------------------------------------------------------------------------------

dist = [influx_d18O_dist, influx_D17O_dist, influx_dD_dist, lake_d18O_dist, lake_D17O_dist, lake_dD_dist, prec_d18O_dist, prec_D17O_dist, prec_dD_dist, temp_dist]
variable = ["δ¹⁸O, influx (‰)", 'Δ′¹⁷O, influx (per meg)', 'δ²H, influx (‰)', "δ¹⁸O, lake (‰)", 'Δ′¹⁷O, lake (per meg)', 'δ²H, lake (‰)', 
            "δ¹⁸O, evaporation flux-weighted precipitation (‰)", 'Δ′¹⁷O, evaporation flux-weighted precipitation (‰)', 'δ²H, evaporation flux-weighted precipitation (‰)',
            'Temperature (°C)']
dist_name = ['influx_d18O_dist', 'influx_D17O_dist', 'influx_dD_dist', 'lake_d18O_dist', 'lake_D17O_dist', 'lake_dD_dist', 'prec_d18O_dist', 'prec_d17O_dist', 'prec_dD_dist', 'temp_dist']


for d, v, d_n in zip(dist, variable, dist_name):

    # Print summary statistics
    
    print("\nvariable: \t", v)
    print("\nmean output: \t", np.nanmean(d))
    print("median output: \t", np.median(d))
    print("stdev output: \t", np.nanstd(d))
    print("25th perc output: \t", np.percentile(d, 25))
    print("75th perc output: \t", np.percentile(d, 75))
    print("IQR (75-25): \t", np.percentile(d, 75)-np.percentile(d, 25))
    print("minimum output: \t", np.min(d))
    print("maximum output: \t", np.max(d))
    print("number of simulations: \t", sim)
    
    # Plot output histograms
    
    binwidth = (max(d) - min(d))/20
    plt.hist(d, color='gray', edgecolor = "black", bins=np.arange(min(d), max(d) + binwidth, binwidth))
    plt.ylabel('Count')
    plt.xlabel(v)
    # plt.savefig('{path}'+v+"_hist.png", bbox_inches="tight", dpi=600)
    plt.show()
    
#%% Define mass balance functions

# Mass balance equations in the triple oxygen space to calculate for humidity and Xe

def calc_hum_17(vars, lake_d18O_, lake_D17O_, temp_, influx_d18O_, influx_D17O_, atm_d18O_, atm_d17O_, n_):
    h_, x_,  = vars
    
    # Calculate equilibrium fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_17_ = lbf.fractionation_factor_d17O(a_18_)
    
    a_k_18_0 = 1.0285 # Kinetic fractionation at pure molecular diffusion (Merlivat 1978)
    a_k_17_0 = 1.0146 # Kinetic fractionation at pure molecular diffusion (Barkan and Luz 2007)
    
    a_k_18 = a_k_18_0**n_
    a_k_17 = a_k_17_0**n_

    # Calculate d17O from measured D17O data
    influx_d17O_ = influx_D17O_/1000 + 0.528*lbf.lnrz_d(influx_d18O_) # get influx d17O from D17O and linearized d18O
    i_17 = 1000*(np.exp(influx_d17O_/1000)-1) # delinearize influx d17O
    
    lake_d17O_ = lake_D17O_/1000 + 0.528*lbf.lnrz_d(lake_d18O_) # get lake d17O  from D17O and linearized d18O
    l_17 = 1000*(np.exp(lake_d17O_/1000)-1) # delinearize lake d17O

    # Convert d values to R
    influx_R18O = lbf.convert_d_to_R(influx_d18O_) 
    influx_R17O = lbf.convert_d_to_R(i_17) 
    
    lake_R18O = lbf.convert_d_to_R(lake_d18O_)
    lake_R17O = lbf.convert_d_to_R(l_17)
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation 
    # is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    atm_R18 = lbf.convert_d_to_R(atm_d18O_) 
    atm_R17 = lbf.convert_d_to_R(atm_d17O_)
    
    # Equation from Surma (2018), Voigt (2021), Passey and Levin (2021)
    eq1 = ((a_18_*a_k_18*(1-h_)*influx_R18O + a_18_*x_*h_*atm_R18))/(a_18_*a_k_18*(1-h_)*(1-x_)+x_) - lake_R18O
    eq2 = ((a_17_*a_k_17*(1-h_)*influx_R17O + a_17_*x_*h_*atm_R17))/(a_17_*a_k_17*(1-h_)*(1-x_)+x_) - lake_R17O

    return np.array([eq1, eq2])

# Mass balance equations in the 18O-2H space to calculate for humidity and Xe

def calc_hum_18(vars, lake_d18O_, lake_dD_, temp_, influx_d18O_, influx_D_, atm_d18O_, atm_dD_, n_):
    h_, x_,  = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_D_ = lbf.fractionation_factor_dD(temp_) 
    
    a_k_18_0 = 1.0285 # Kinetic fractionation factor at pure molecular diffusion (Merlivat 1978)
    a_k_D_0 = 1.0251 # Kinetic fractionation factor at pure molecular diffusion (Merlivat 1978)
    
    a_k_18 = a_k_18_0**n_
    a_k_D = a_k_D_0**n_

    # Convert d values to R
    influx_R18O = lbf.convert_d_to_R(influx_d18O_) 
    influx_RD = lbf.convert_d_to_R(influx_D_) 
    
    lake_R18O = lbf.convert_d_to_R(lake_d18O_)
    lake_RD = lbf.convert_d_to_R(lake_dD_)
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation 
    # is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    atm_R18O = lbf.convert_d_to_R(atm_d18O_) 
    atm_RD = lbf.convert_d_to_R(atm_dD_)

    # Equation from Surma (2018), Voigt (2021), Passey and Levin (2021)
    eq1 = ((a_18_*a_k_18*(1-h_)*influx_R18O + a_18_*x_*h_*atm_R18O))/(a_18_*a_k_18*(1-h_)*(1-x_)+x_) - lake_R18O
    eq2 = ((a_D_*a_k_D*(1-h_)*influx_RD + a_D_*x_*h_*atm_RD))/(a_D_*a_k_D*(1-h_)*(1-x_)+x_) - lake_RD

    return np.array([eq1, eq2])

# Equation to reconstruct the d18O of unevaporated catchment precipitation (rucp). Equation from Passey and Ji (2019)

def rucp(D17O_ref, D17O_lake, m_ref, d18O_lake, m_lake):
    rucp = (D17O_ref/1000-D17O_lake/1000+((m_lake-m_ref)*d18O_lake))/(m_lake-m_ref)
    
    return rucp

# Function to remove improbable data for h and Xe (less than zero or more than one)
def remove_improbable(data):
    new_list = [i for i in data if (i>=0 and i<=1)]
    return new_list

# Function to remove improbable d18O rucp (estimated from literature, can be changed)
def remove_improbable_rucp(data): 
    new_list = [i for i in data if (i>=-30 and i<=-10)]
    return new_list

#%% Monte Carlo analysis to calculate for the isotopic composition of atmospheric moisture, humidity, Xe

# Initialize output arrays

atm_d18O_dist = []
atm_d17O_dist = []
atm_D17O_dist = []
atm_dD_dist = []
atm_d_exc_dist = []


a_18_dist = []
a_17_dist = []
a_D_dist = []

x_dist = []
h_dist = []

x_17_dist = []
h_17_dist = []

# Input initial humidity (h), and Xe (x_1)

hum = 0.62 # Values obtained from prior work (Custado, et al. 2025)
x_1 = 0.38

initial_guess = [hum, x_1]

## Run simulations

sim2 = len(temp_dist)

for ind in np.arange(0,sim2,1):
    print(ind)
    
    # Calculate fractionation factors
    
    a_18 = lbf.fractionation_factor_d18O(temp_dist[ind]) 
    a_18_dist.append(a_18)

    a_17 = lbf.fractionation_factor_d17O(a_18)
    a_17_dist.append(a_17)

    a_D = lbf.fractionation_factor_dD(temp_dist[ind])
    a_D_dist.append(a_D)

    ep_18 = (a_18 - 1)*1000
    ep_17 = (a_17 - 1)*1000
    ep_D = (a_D - 1)*1000
    
    # Calculate isotopic composition of atmospheric moisture
    x_18O = lbf.isotope_atm(prec_d18O_dist[ind], ep_18, 1) 
    atm_d18O_dist.append(x_18O)
    
    x_17O = lbf.isotope_atm(prec_d17O_dist[ind], ep_17, 1) 
    atm_d17O_dist.append(x_17O)
    
    x_D17O = lbf.lnrz_d(x_17O)-0.528*lbf.lnrz_d(x_18O)
    atm_D17O_dist.append(1000*x_D17O)

    x_D = lbf.isotope_atm(prec_dD_dist[ind], ep_D, 1) 
    atm_dD_dist.append(x_D)
    
    x_d_exc = x_D - 8*x_18O
    
    atm_d_exc_dist.append(x_d_exc)

for ind in np.arange(0,sim2,1):
    print(ind)
    
    # Calculate h and Xe in triple oxygen space

    roots = opt.fsolve(calc_hum_17, [initial_guess], args=(lake_d18O_dist[ind], lake_D17O_dist[ind], temp_dist[ind], influx_d18O_dist[ind], influx_D17O_dist[ind], atm_d18O_dist[ind], atm_d17O_dist[ind], 0.5))
    h_17_dist.append(roots[0])
    x_17_dist.append(roots[1])
    
    # Calculate h and Xe in d18-dD space
    
    roots = opt.fsolve(calc_hum_18, [initial_guess], args=(lake_d18O_dist[ind], lake_dD_dist[ind], temp_dist[ind], influx_d18O_dist[ind], influx_dD_dist[ind], atm_d18O_dist[ind], atm_dD_dist[ind], 0.5))
    h_dist.append(roots[0])
    x_dist.append(roots[1])

# Compile calculated distributions

dist = [a_18_dist, a_17_dist, a_D_dist, atm_d18O_dist, atm_d17O_dist, atm_D17O_dist, atm_dD_dist, atm_d_exc_dist, remove_improbable(x_dist), remove_improbable(h_dist), remove_improbable(x_17_dist), remove_improbable(h_17_dist)]
variable = ["α(eq) δ¹⁸O", "α(eq) δ¹⁷O", "α(eq) δ²H", "δ¹⁸O, atmosphere (‰)", 'δ¹⁷O, atmosphere (‰)', 'Δ''¹⁷O, atmosphere (per meg)', 'δ²H, atmosphere (‰)', 'd-excess, atmosphere (‰)', 'Xₑ, δ¹⁸O-δ²H', 'h, δ¹⁸O-δ²H', 'Xₑ, Triple oxygen', 'h, Triple oxygen']
dist_name = ['a_18_dist', 'a_17_dist', 'a_D_dist', 'atm_d18O_dist', 'atm_d17O_dist', 'atm_D17O_dist', 'atm_dD_dist', 'atm_d_exc_dist', 'x_dist', 'h_dist', 'x_17_dist', 'h_17_dist']

for d, v, d_n in zip(dist, variable, dist_name):

    # Print summary statistics

    print("\nvariable: \t", v)
    print("\nmean output: \t", np.nanmean(d))
    print("median output: \t", np.median(d))
    print("stdev output: \t", np.nanstd(d))
    print("25th perc output: \t", np.percentile(d, 25))
    print("75th perc output: \t", np.percentile(d, 75))
    print("IQR (75-25): \t", np.percentile(d, 75)-np.percentile(d, 25))
    print("minimum output: \t", np.min(d))
    print("maximum output: \t", np.max(d))
    print("number of simulations: \t", sim)
    
    # Plot output histograms
    
    binwidth = (max(d) - min(d))/20
    plt.hist(d, color='gray', edgecolor = "black", bins=np.arange(min(d), max(d) + binwidth, binwidth))
    plt.ylabel('Count')
    plt.xlabel(v)
    # plt.savefig('{path}'+v+"_hist.png", bbox_inches="tight", dpi=600)
    plt.show()

#%% Monte Carlo analysis to calculate theta lake and rucp

# Input calculated values and uncertainties (unc) for humidity (h) and Xe (x) to generate evaporation trajectory 
# mwl refers to Global Meteoric Water Line 

# Values from the triple oxygen calculations

# Median values
h_calc = 0.590 # median
x_calc = 0.267 # median
mwl_D17O = 32 # Passey and Ji (2019)

# Uncertainties
h_calc_unc = 0.399 # IQR
x_calc_unc = 0.077 # IQR
mwl_D17O_unc = 15 # Passey and Ji (2019)

# Input lake isotopic values from data
lake_d18O = in_iso.loc[in_iso['variable']=='lake_d18O']['value']
lake_D17O = in_iso.loc[in_iso['variable']=='lake_D17O']['value']

# Initialize distribution arrays for analysis (sim = number of simulations similar to previous section)
h_calc_dist = np.random.default_rng().normal(loc=h_calc, scale=h_calc_unc, size=sim)
x_calc_dist = np.random.default_rng().normal(loc=x_calc, scale=x_calc_unc, size=sim)
mwl_D17O_dist = np.random.default_rng().normal(loc=mwl_D17O, scale=mwl_D17O_unc, size=sim)

# Initialize output arrays
lake_theta_array = []
rucp_array = []

influx_d17O_array = []
lake_d17O_array = []

for ind in np.arange(0,sim2,1):
    
    # Calculate theta_lake by using (1) inflow and lake isotopic data or (2) equation from Passey and Ji (2019)
    # Comment out which method to not use
    
    #### Method (1): using inflow and lake isotopic data ####
    influx_d17O_ = influx_D17O_dist[ind]/1000 + 0.528*lbf.lnrz_d(influx_d18O_dist[ind]) # get d17O influx from D17O and d18O
    influx_d17O_array.append(influx_d17O_)
    
    lake_d17O_ = lake_D17O_dist[ind]/1000 + 0.528*lbf.lnrz_d(lake_d18O_dist[ind]) # get d17O lake from D17 and d18O
    lake_d17O_array.append(lake_d17O_)

    y = [influx_d17O_, lake_d17O_]
    x = [lbf.lnrz_d(influx_d18O_dist[ind]), lbf.lnrz_d(lake_d18O_dist[ind])]
    
    m_lake = np.polyfit(x,y,1)
    m_lake1 = m_lake[0]
    lake_theta_array.append(m_lake1)
    
    #### Method (2): using equation from Passey and Ji (2019) and Katz et al. (2023) ####
    # Comment out lines not in use
    # m_lake1 = 1.9856*(lake_D17O_dist[ind]/1000**3) + 0.5730*(lake_D17O_dist[ind]/1000**2) + 0.0601*lake_D17O_dist[ind]/1000 + 0.5236 # Passey and Ji (2019)
    # m_lake1 = 9.8941*(lake_D17O_dist[ind]/1000**3) + 1.2759*(lake_D17O_dist[ind]/1000**2) + 0.0580*lake_D17O_dist[ind]/1000 + 0.5229 # Full humidity, 0.3-0.9, Katz et al. (2023)
    # m_lake1 = 1.9600*(lake_D17O_dist[ind]/1000**3) + 0.5340*(lake_D17O_dist[ind]/1000**2) + 0.0599*lake_D17O_dist[ind]/1000 + 0.5237 # Low humidity, 0.7-0.9, Katz et al. (2023)
    # m_lake1 = 40.9869*(lake_D17O_dist[ind]/1000**3) + 1.9363*(lake_D17O_dist[ind]/1000**2) + 0.0701*lake_D17O_dist[ind]/1000 + 0.5215 # High humidity, 0.3-0.7, Katz et al. (2023)

    lake_theta_array.append(m_lake1)

    r = rucp(32, lake_D17O.values, 0.528, lbf.lnrz_d(lake_d18O.values), m_lake1)
    
    # Delinearize output rucp
    r = 1000*(np.exp(r/1000)-1)
    rucp_array.append(r[0])

# Print summary statistics
dist = [lake_theta_array, remove_improbable_rucp(rucp_array), influx_d17O_array, lake_d17O_array]
variable = ["θ lake", "δ'¹⁸O rucp (‰)", "δ'17O, influx (‰)", "δ'17O, lake (‰)"]

for d, v in zip(dist[0:2], variable[0:2]):

    
    print("\nvariable: \t", v)
    print("\nmean output: \t", np.nanmean(d))
    print("median output: \t", np.median(d))
    print("std output: \t", np.nanstd(d))
    print("25 perc output: \t", np.percentile(d, 25))
    print("75 perc output: \t", np.percentile(d, 75))
    print("IQR: \t", np.percentile(d, 75) - np.percentile(d, 25))
    print("minimum output: \t", np.min(d))
    print("maximum output: \t", np.max(d))
    print("number of simulations: \t", sim)
    
    plt.hist(d, color='gray', edgecolor = "black")
    plt.ylabel('Count')
    plt.xlabel(v)
    # plt.savefig('{path}'+v+"_hist.png", bbox_inches="tight", dpi=600)
    plt.show()

#%% Save output data (mean, stdev, median, IQR) in a csv file
all_data = {
    "h_d18O_mean": [np.nanmean(remove_improbable(h_dist)), np.nanstd(remove_improbable(h_dist))],
    "h_d17O_mean": [np.nanmean(remove_improbable(h_17_dist)), np.nanstd(remove_improbable(h_17_dist))],
    "h_d18O_median": [np.median(remove_improbable(h_dist)), (np.percentile(remove_improbable(h_dist), 75)-np.percentile(remove_improbable(h_dist), 25))],
    "h_d17O_median": [np.median(remove_improbable(h_17_dist)), (np.percentile(remove_improbable(h_17_dist), 75)-np.percentile(remove_improbable(h_17_dist), 25))],

    "x_d18O_mean": [np.nanmean(remove_improbable(x_dist)), np.nanstd(remove_improbable(x_dist))],
    "x_d17O_mean": [np.nanmean(remove_improbable(x_17_dist)), np.nanstd(remove_improbable(x_17_dist))],
    "x_d18O_median": [np.median(remove_improbable(x_dist)), (np.percentile(remove_improbable(x_dist), 75)-np.percentile(remove_improbable(x_dist), 25))],
    "x_d17O_median": [np.median(remove_improbable(x_17_dist)), (np.percentile(remove_improbable(x_17_dist), 75)-np.percentile(remove_improbable(x_17_dist), 25))],
    
    "atm_d18O_mean": [np.nanmean(atm_d18O_dist), np.nanstd(atm_d18O_dist)],
    "atm_d17O_mean": [np.nanmean(atm_d17O_dist), np.nanstd(atm_d17O_dist)],
    "atm_dD_mean": [np.nanmean(atm_dD_dist), np.nanstd(atm_dD_dist)],
    "atm_D17O_mean": [np.nanmean(atm_D17O_dist), np.nanstd(atm_D17O_dist)],
    "atm_d_exc_mean": [np.nanmean(atm_d_exc_dist), np.nanstd(atm_d_exc_dist)],

    "atm_d18O_median": [np.median(atm_d18O_dist), (np.percentile((atm_d18O_dist), 75)-np.percentile((atm_d18O_dist), 25))],
    "atm_d17O_median": [np.median(atm_d17O_dist), (np.percentile((atm_d17O_dist), 75)-np.percentile((atm_d17O_dist), 25))],
    "atm_dD_median": [np.median(atm_dD_dist), (np.percentile((atm_dD_dist), 75)-np.percentile((atm_dD_dist), 25))],
    "atm_D17O_median": [np.median(atm_D17O_dist), (np.percentile((atm_D17O_dist), 75)-np.percentile((atm_D17O_dist), 25))],
    "atm_d_exc_median": [np.median(atm_d_exc_dist), (np.percentile((atm_d_exc_dist), 75)-np.percentile((atm_d_exc_dist), 25))],

}

# convert to DataFrame (transpose so each variable is a row)
all_data_df = pd.DataFrame(all_data, index=["Value", "Uncertainty"]).T.reset_index()
all_data_df.columns = ["variable", "value", "uncertainty"]

# save to CSV
# all_data_df.to_csv(r'{path}'+"_calc_outputs.csv")




